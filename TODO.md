# EMST optimization TODO

Opportunities identified in `include/panna/emst.hpp`. Two groups:
[implementation optimizations](#implementation-optimizations) (how the work is
done) and [algorithmic changes](#algorithmic-changes) (what work gets done).

# Implementation optimizations

Roughly in priority order.

## 1. Serial collector is the likely scaling bottleneck
In `find_tree` (lines ~748–861) and `find_tree_mutual_reachability_distance`
(~948–1019), workers run in parallel but all merging happens in a single
collector loop. For *every* partial received it does O(n) work:
- `std::vector<Edge> tree( running_result.read()->tree )` — full tree copy
- `kruskal_merge(...)` with an internal `union_find.reset()` (O(n))
- `complete_arbitrarily(...)` — which itself does `std::sort(forest)` every call
- `stopping_condition(tree, ...)` — takes the tree by value

This is `O(max_repetitions · n log n)` of strictly serial work per prefix per
rehash. Profile first to confirm the worker-vs-collector split, then attack.

## 2. `complete_arbitrarily` is called per partial and re-sorts every time
Called purely to test the stopping condition; `std::sort(forest)` dominates.
Options: only complete/sort when `tree.size()` is close to `num_data-1`; insert
completion edges in sorted order and merge instead of full re-sort; or maintain
weight incrementally so the stopping check doesn't need a fully completed sorted
tree.

## 3. `.at()` bounds-checking in inner loops
The merge kernels (`kruskal_new_edges`, `kruskal_merge`) and
`CoreDistances::do_update` / `core_distance` use `.at()` everywhere — the
innermost loops over all candidate edges. Switch to `operator[]` / iterators
(indices are already provably in range) to drop a branch per access.

## 4. Heavy per-repetition deep copies
Each worker copies state out of the `Billboard` snapshot every repetition:
- `worker_fun`: copies `local_tree`, `filter`, and a second DSU `dsu`.
- `worker_fun_mutual_reachability`: copies the entire `CoreDistances`
  (`num_points · num_neighbors` pairs) per repetition, and again in the collector.

Share the immutable snapshot read-only and only copy the mutable delta, or make
`CoreDistances` cheap to snapshot (COW / `shared_ptr` to the neighbor array,
since the read path doesn't mutate it).

## 5. `update_tree` re-allocates and full-sorts from scratch
Builds a fresh `std::vector<MREdge> all`, pushes tree+updates, then `std::sort`s
the whole thing and allocates a new DSU each call. Since `tree` is already
(nearly) sorted, merging two sorted runs beats a full sort; the `all` buffer and
DSU could be reused across calls (thread-local scratch). The commented-out dead
pruning block also suggests `updates` may accumulate duplicates over time (see
the `OPTIMIZE` comment).

---

**Suggested order:** #2/#1 (restructure the serial collector so completion + sorting isn't redone per
partial), and #4 (kill the `CoreDistances` copies in the MR path). Start by
profiling a representative run to confirm the collector-vs-worker split before
the larger #1/#5 refactors.

# Algorithmic changes

These change *what work gets done*, not just how fast each operation runs.

## A1. Component contraction (Borůvka-style super-nodes) — biggest structural win
The DSU `filter` only *skips* same-component pairs at query time
(`filter.cfind`), but the index still stores and hashes every original point for
the whole run. As the tree fills, ever more collisions are enumerated only to be
discarded.

Instead, physically contract confirmed components into representatives: once a
component is internally confirmed, its interior points never need to be hashed
against anything again (keep one or a few boundary representatives per
component). Bucket sizes and collision enumeration cost then shrink
geometrically as the tree grows — directly attacking the regime changelog
entry #5 was already fighting ("scales quadratically with the number of
components").

## A2. Switch algorithms when few components remain
Cost is lopsided: long prefixes cheaply find the many *short* MST edges; the last
handful of *long* edges connecting far-apart clusters need short prefixes that
generate enormous collision sets. Once `num_connected_components` drops below a
threshold (a few hundred), abandon LSH for the tail and go direct:
- compute distances among component representatives directly (small set), or
- a single exact / dual-tree pass on representatives.

`scripts/emst.py` already has a whole-dataset algorithm selector; this would be
an *intra-run* selector.

## A3. Adaptive prefix selection instead of `for prefix = K … 1`
Many prefix levels contribute zero useful edges (all collisions intra-component
or duplicate). Drive prefix choice from the distance distribution of the current
unconfirmed frontier: pick the prefix whose expected collision distance matches
the weights still to confirm, and skip / binary-search over empty levels. Fewer
wasted full repetition sweeps.

## A4. Early termination within a prefix
Each prefix currently consumes all `max_repetitions`. Track useful-edge yield per
repetition and bail out early when consecutive repetitions add nothing. Composes
with A3 to cut redundant index scans.

## A5. Tighten the stopping condition (fewer repetitions overall)
Highest-leverage theoretical lever — it directly controls how many repetitions
are needed, and it gates everything else. Confirming edges earlier → fewer
repetitions → less work everywhere.

### What the current condition proves
`stopping_condition` (lines ~1181–1219) walks tree edges in ascending weight and
accumulates a failure budget:

    prob += fail_probability(w, i, j)   // i = prefix/concat, j = reps
    confirm edge while prob <= delta

`fail_probability(w,i,j)` is the probability a pair at distance `w` was *missed*
(`(1 - p(w)^K)^L`-style); it *increases* with `w` (long edges are hard to find),
so small edges are cheap to confirm and the cumulative union bound crosses
`delta` at index `idx`. Result: w.p. >= 1-delta the `confirmed_edges` smallest
edges are genuine MST edges (a subforest of OPT).

`find_tree` (lines ~800–805) then forms:

    lower_bound = confirmed_weight + edges_to_confirm * heaviest_confirmed_edge
    stop when total_weight <= (1+epsilon) * lower_bound

Argument: contract the confirmed forest into super-nodes; OPT still needs
`edges_to_confirm` more crossing edges, each (conditioned on confirmation) of
weight >= `heaviest_confirmed_edge` (anything shorter would have been found). As
reps grow, `total_weight` falls and `lower_bound` rises until within epsilon.

### Where it is loose, and how to tighten (priority order)

1. **Use the confirmation radius instead of `heaviest_confirmed_edge`
   (biggest free win, local change).** At the break point there is leftover
   budget `delta - prob > 0` that is thrown away. The honest multiplier is the
   largest distance still provably free of missed edges given that budget:

       r* = max{ d : fail_probability(d, i, j) <= (delta - prob) / edges_to_confirm }

   `fail_probability` is monotone increasing and invertible in `d` (binary-search
   it, or invert the closed form), so `r* >= heaviest_confirmed_edge`, often
   strictly. Replacing the multiplier with `r*` gives a strictly larger,
   still-valid lower bound → earlier stop, no change to the guarantee. Local to
   `stopping_condition` + the lower-bound formula.

2. **Per-component radius instead of one global value.** A single
   `heaviest_confirmed_edge` / `r*` is uniform across all unconfirmed cuts, but
   after rehashing different regions have different effective `(i,j)` histories
   (`fail_probability` already multiplies in `rehash_history_hashers`), and dense
   clusters get confirmed to a much larger radius than sparse ones. Sum
   `Σ_components r*_c` instead of `count * r*_global`. Shares per-component state
   with A1/A7.

3. **`complete_arbitrarily` pollutes `total_weight`.** When the tree is not truly
   connected by found edges, `complete_arbitrarily` (line 784) injects real-but-
   heavy connecting edges that inflate `total_weight`, making the ratio test
   harder and stopping *later* (safe but slow). Fix: do not fold arbitrary
   completion weights into the `total_weight` used in the ratio — bound the
   unconfirmed portion consistently on both sides — and/or lower the upper bound
   legitimately by finding genuinely short tail edges (A2). The tail is exactly
   where the ratio test stalls.

4. **Union bound is the conservative core.** `prob += fp` is `P(any confirmed
   edge wrong) <= Σ fp`; across reps survival is already multiplicative (good),
   across edges it is additive. For small delta this is near-tight, so low
   priority — but note the budget is implicitly spent on *both* confirmation and
   the radius claim, which are the same "a short edge was missed" event.
   Unifying them under one delta (no double counting) is cleaner and slightly
   tighter.

5. **Data-dependent bounds (phase 2).** `fail_probability` is worst-case over the
   unknown distance distribution. We actually observe collision counts per
   repetition/prefix; an empirical-Bernstein bound on the per-distance collision
   rate can be markedly tighter on benign (i.e. most real) data. This is the
   "adaptive / sequential testing" path — more effort, attacks the repetition
   constant directly. Follow-up to 1/2.

### Correctness caveat
Every change above *relaxes* when we stop, so each must preserve the >= 1-delta
guarantee. Validate empirically against `exact_tree` (line 501) on small/medium
instances: confirm realized approximation ratio stays <= 1+epsilon and failure
rate <= delta across many seeds *before* trusting on large data. Gate #1 behind
that check first (pure win), then layer #2 and #3.

## A6. Mutual-reachability path: compute core distances up front
The MR/HDBSCAN path intertwines core-distance refinement with MST construction,
forcing the "stash possibly-useful edges because core distances might drop later"
machinery (`non_tree_edges`, `can_improve`, repeated re-sorts in `update_tree`).
Simpler and likely faster: compute accurate k-NN / core distances once (via the
LSH index, NN-descent, or the `fast_hdbscan` machinery already in the flake),
then run a *single* EMST on the mutual-reachability graph with fixed weights. No
edge gets cheaper later, so the stash-and-recheck logic disappears and the plain
`find_tree` path can be reused.

## A7. Cache the best edge per component pair
Across repetitions and prefixes the same inter-component pair is rediscovered and
its distance recomputed. Maintain the current best candidate edge per pair of
*active components* (naturally tied to the contraction in A1) to avoid
recomputation and feed the Borůvka "min outgoing edge per component" step.

---

**Ranking:** A1 (component contraction) and A2 (switch for the tail) are the
structural game-changers — they target the documented quadratic-in-components
pain. A6 is a clean simplification *and* speedup for HDBSCAN. A5 is the
highest-leverage theoretical lever for fewer repetitions everywhere. A3/A4/A7 are
incremental refinements that compose with A1.

---

# Parallel code reorganization (refactor plan)

This is a concrete, staged plan to restructure the threading in `find_tree`
(~697–894) and `find_tree_mutual_reachability_distance` (~896–1045). It folds in
implementation items #1, #2, #5, #6, #7 and creates the seams that A3/A4 need.

## The diagnosis it addresses

Today both `find_tree` paths fuse three responsibilities into one per-partial
serial collector loop, wrapped in **fork-join per prefix**:

1. **reduce** — `kruskal_merge` a worker partial into the global tree;
2. **decide** — `complete_arbitrarily` + `stopping_condition`;
3. **publish** — `running_result.update(...)`.

Costs per partial: an O(n) tree *copy* (761/961), an O(n) `kruskal_merge` with a
freshly-allocated DSU (762/771), an O(n log n) `complete_arbitrarily` sort (784),
and an O(n) `stopping_condition` taken **by value** (1181/1221). Workers
additionally deep-copy the whole snapshot every repetition (592–594 copy tree +
two DSUs; 653–667 copy tree + `CoreDistances` + DSU). Threads are created
`max_threads × prefixes × rehashes` times.

**Correctness boundary to respect:** `kruskal_merge` (98) intentionally re-runs
Kruskal over `old + new` from a `reset()` DSU because a newly-arrived light edge
may displace a heavier edge already in the tree. So the per-partial merge stays
inherently O(n) *unless* we adopt a dynamic-MST structure (link-cut tree) — out
of scope here. This plan removes the avoidable copies/allocs and the avoidable
O(n log n), restructures the threading, and leaves the asymptotic shrink to A1
(component contraction reduces the effective n).

## Phase 0 — Verification harness (do first, no production change)
Before touching anything, lock in a behavioural baseline so every later phase is
provably non-regressing:
- A test that runs `find_tree` and `find_tree_mutual_reachability_distance`
  against `exact_tree` (line 501) on several small/medium instances and seeds,
  asserting realized ratio ≤ 1+epsilon and tree weight matches the current
  implementation’s output bit-for-bit (these phases are behaviour-preserving;
  only Phase 2’s cadence change is observable, and only in *when* we stop).
- Capture a wall-clock + `count_distances`/`count_collisions` baseline via
  `PANNA_EMST_THREADS=1` and at default fan-out on one representative dataset.
- **Profile to confirm the collector-vs-worker split** (item #1) so we attack the
  real bottleneck, not the assumed one.

## Phase 1 — Collector owns the tree; reuse scratch buffers
*Behaviour-preserving; kills the per-partial copies and allocations (items #1, #6).*
- Introduce a collector-local `struct ReducerState { std::vector<Edge> tree; DSU
  union_find; DSU completion_filter; std::vector<Edge> merge_scratch; }`
  constructed **once** before the receive loop, not per partial.
- Replace `std::vector<Edge> tree( running_result.read()->tree )` (761/961) with
  the owned `state.tree`; merge the incoming `update` into `merge_scratch` and
  `swap` — no fresh `DSU filter(num_data)` per iteration (762/952), `reset()` the
  persistent ones instead.
- Publishing to the `Billboard` still needs a snapshot the workers can read; keep
  one copy *there* (the unavoidable one) but the collector’s working tree is no
  longer copied in.
- MR path: same, plus carry one owned `CoreDistances` instead of
  `CoreDistances core_distances(snapshot->neighborhoods)` (962) every partial.
- Verify against Phase 0 (identical output expected).

## Phase 2 — Put `decide` on its own cadence
*Item #2; the one observable change (we may stop a few partials later/earlier),
gated by the Phase 0 ratio check.*
- `complete_arbitrarily` + `stopping_condition` currently run every partial. Gate
  them behind a cheap necessary condition computed from the *real* found edges
  (no completion): only attempt when the found forest’s
  `union_find.num_connected_components()` is at/under a small threshold (the
  regime where stopping can plausibly fire), or every `k` partials as a fallback.
- Change both `stopping_condition` overloads (1181, 1221) to take
  `const std::vector<Edge>&` (item #7); thread the tree through by reference.
- Optional within this phase: avoid the full `std::sort` in `complete_arbitrarily`
  by inserting completion edges into a pre-sorted position / merging (item #2).
- Verify ratio ≤ 1+epsilon and failure rate ≤ delta across seeds.

## Phase 3 — Read-only snapshots in workers
*Item #5.*
- `Billboard::read()` already returns `shared_ptr<const T>`. Hold it for the whole
  repetition and call `filter.cfind` through it instead of copying the DSU
  (592–594). The only mutable per-repetition state is the worker’s Kruskal scratch
  (`output`, local `dsu`) — keep those worker-local.
- MR path: put the `CoreDistances` neighbor array behind a `shared_ptr` (COW) so a
  snapshot is a pointer bump, not a `num_points × num_neighbors` copy (653–654).
  Workers that need to *probe* `can_improve` read through the shared pointer; the
  collector is the sole mutator.
- This is only safe once the collector is the single owner/mutator (Phase 1),
  which is why it comes after.

## Phase 4 — Persistent worker pool across prefixes *and* rehashes
*Removes thread churn; enables overlap across prefixes.*
- New `struct WorkItem { size_t prefix; size_t repetition; uint32_t epoch; };`
  where `epoch` is the rehash generation.
- Spawn `max_threads` workers **once** at the top of `find_tree`, looping on a
  single `Channel<WorkItem> work` until close (shutdown sentinel). The worker body
  is today’s `worker_fun` minus its outer prefix loop.
- Main thread becomes reducer+driver: enqueue work, receive deltas, integrate
  (Phase 1 state), decide (Phase 2 cadence), publish, enqueue more.
- **Rehash quiescence barrier (correctness-critical):** `table.rehash(...)`
  (876/1034) mutates the shared `Index` non-atomically while
  `search_pairs_different_groups` reads it. Today that’s safe only because all
  workers have joined. With a persistent pool: the driver stops enqueuing, waits
  until `outstanding == 0` (enqueued − received) so no worker is inside the index,
  performs the rehash, bumps `epoch`, then resumes. Deltas carry their `epoch`;
  the reducer integrates all epoch-e deltas before the rehash to e+1.
- Verify: identical results to Phase 3; confirm thread-creation count drops to
  `max_threads` total (e.g. via a counter or perf).

## Phase 5 — Extract the driver/policy
*Creates the seam for A3/A4; no behaviour change initially.*
- Pull “which `WorkItem` to enqueue next, when to rehash, when to stop” into a
  `Driver` object. Its initial policy reproduces today’s `for prefix = K…1 { all
  repetitions }` exactly (Phase 0 still passes).
- Once isolated, A3 (adaptive prefix selection) and A4 (early termination within a
  prefix) become local changes to `Driver` — feed it the per-repetition useful
  yield the workers already log (618 `output` size / 624–628 stats) and let it
  skip empty prefix levels or bail a prefix early.

## Phase 6 — (future / only if profiling demands) parallel reduce
*Do not start until Phases 1–5 are in and re-profiled.*
- If the single reducer thread is still the bottleneck after the copies/allocs and
  the decide-cadence are gone, options: pairwise reduction tree over partials, or
  a shared “best edge per active-component pair” map (this *is* A7’s structure and
  composes with A1’s contraction). A correct concurrent DSU is the most
  error-prone piece in this whole plan — gate it behind a clear profiling win.

## Suggested sequencing
0 → 1 → 2 → 3 → 4 → 5, each shippable and verified against Phase 0 independently.
Phases 1–3 are pure constant-factor + allocation wins with no threading-model
change (low risk). Phase 4 is the one with real new synchronization (the rehash
barrier) — land it on its own. Phase 5 is mechanical but unlocks the A3/A4
algorithmic wins. The asymptotic shrink still comes from A1; this plan makes the
machine cheaper per unit work and gives A1/A3/A4 clean places to attach.
