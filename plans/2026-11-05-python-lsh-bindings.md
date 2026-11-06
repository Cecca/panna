---
title: Python bindings for the LSH families
status: done
created: 2026-11-05
completed: 2026-11-05
---

# Python bindings for the LSH families

## Goal

Expose the LSH hash families in `include/panna/lsh/` directly to Python, so
users can build hashers, hash NumPy data, and query collision probabilities
without going through `TrieIndex` or `EMST`.

## Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Families | `Simhash`, `CrossPolytope`, `E2LSH`, `LatticeLSH`, `MinHash`, `MinHash1Bit` |
| Surface | hashing, `collision_probability`, data-dependent fitting, introspection |
| API shape | one Python class per family |
| `K` (concatenations) | a fixed set of instantiations (powers of two up to each family's max), chosen at runtime |
| Output | `hash(data, packed=False)` returns per-concatenation codes; `packed=True` returns one key per repetition |
| Packed keys | `uint32` for bitwise families; fixed-width NumPy void dtype `V{K*sizeof(symbol)}` for the others |
| Seeding | optional `seed=None` argument on constructors and `fit`; when given, it re-seeds the global RNG before sampling |
| Execution | rows are hashed in parallel with OpenMP, with the GIL released |
| Set input (MinHash) | CSR arrays `(indptr, indices)`, plus a convenience that also accepts a scipy CSR matrix or a list of int sequences |
| Fitting (E2LSH, Lattice) | the constructor takes explicit parameters; a static `fit(data, ...)` runs the builder search and returns a hasher |
| Cosine families | input is copied into `UnitNormPoints` (normalized internally); distances are cosine distances |
| Layout | append to `python/wrapper.cpp`; classes exported at the top level of `pypanna` |
| C++ fixes | fix the `minhash.hpp` issues as a prerequisite phase |
| Tests/docs | pytest unit tests and `.pyi` type stubs |

Out of scope: `Tensoring`, the `predicates.hpp` functions (`failure_probability`,
`distance_at_failure_probability`), `E2LSHBallSampler` (added by the
entropy-LSH merge), pickling, README/CHANGELOG updates, and CI test steps.

## Family overview

| Python class | C++ type (Dataset, Distance) | Symbol / value | `K` set | Codes dtype | Packed dtype |
|---|---|---|---|---|---|
| `Simhash` | `Simhash<K, UnitNormPoints, CosineDistance>` | `BitwiseLshValue<K>` | 1, 2, 4, 8, 16, 32 | `uint8` (0/1) | `uint32` |
| `CrossPolytope` | `CrossPolytope<K, UnitNormPoints, CosineDistance>` | `ShortLshValue<K>` | 1, 2, 4, 8, 16 | `int16` | `V{2K}` |
| `E2LSH` | `E2LSH<K, EuclideanPoints, EuclideanDistance>` | `BytewiseLshValue<K>` | 1, 2, 4, 8, 16 | `int8` | `V{K}` |
| `LatticeLSH` | `LatticeLSH<K, EuclideanPoints, EuclideanDistance>` | `LongLshValue<K>` | 1, 2, 4, 8, 16 | `int64` | `V{8K}` |
| `MinHash` | `Minhash<K, SparseSets>` | `IntLshValue<K>` | 1, 2, 4, 8, 16 | `int32` | `V{4K}` |
| `MinHash1Bit` | `Minhash1Bit<K, SparseSets>` | `BitwiseLshValue<K>` | 1, 2, 4, 8, 16, 32 | `uint8` (0/1) | `uint32` |

`BitwiseLshValue` limits `K` to 32. The symbol-based families have no hard
limit, so they are capped at 16 to keep binary size and compile time in check.
That makes 32 template instantiations in total.

## Proposed Python API

```python
import pypanna as pp

# Cosine families
h = pp.Simhash(dimensions=128, k=16, repetitions=64, seed=None)
h = pp.CrossPolytope(dimensions=128, k=4, repetitions=64,
                     estimation_repetitions=1024, estimation_eps=5e-3, seed=None)

# Euclidean families: explicit parameters...
h = pp.E2LSH(dimensions=128, k=8, repetitions=64, quantization_width=4.0, seed=None)
h = pp.LatticeLSH(dimensions=128, k=4, repetitions=64, scaling_factor=2.0,
                  offset=None, seed=None)   # offset: (d,) array, default zeros
# ...or fitted on data (static method; returns a hasher)
h = pp.E2LSH.fit(data, k=8, repetitions=64, delta=0.1, seed=None)
h = pp.LatticeLSH.fit(data, k=4, repetitions=64, delta=0.1, seed=None)

# Jaccard families
h = pp.MinHash(k=4, repetitions=64, seed=None)
h = pp.MinHash1Bit(k=16, repetitions=64, seed=None)

# Common surface
codes = h.hash(X)                # (n, repetitions, k), family dtype
keys  = h.hash(X, packed=True)   # (n, repetitions), uint32 or V{nbytes}
h.hash(x)                        # 1-D input -> (repetitions, k) / (repetitions,)
p = h.collision_probability(d)   # float -> float, ndarray -> ndarray
h.k, h.repetitions, h.dimensions # read-only properties (no dimensions for MinHash)
h.quantization_width             # E2LSH only
h.scaling_factor, h.offset       # LatticeLSH only
str(h)                           # e.g. "E2LSH(k=8, repetitions=64, quantization=4.0)"

# MinHash input
h.hash(indptr, indices, packed=False)  # CSR arrays of uint32 tokens
h.hash_sets(obj, packed=False)         # scipy.sparse CSR matrix or list of int sequences
```

Error handling: an unsupported `k` raises `ValueError` and lists the allowed
values. So do a dimension mismatch, a non-positive quantization width or
scaling factor, and an `offset` of the wrong length. `fit` on empty data
raises `ValueError` (translated from the C++ `std::invalid_argument`).

## Implementation plan

### Phase 0: fix `minhash.hpp` (prerequisite)

1. `Minhash1Bit::hash`: initialize `cur` to `0`. It is currently read
   uninitialized on the first repetition.
2. Add `static constexpr size_t get_concatenations()` and `get_repetitions()`
   to `Minhash1Bit`, and `get_repetitions()` to `Minhash`. (`Minhash` already
   has `get_concatenations()`.)
3. Make `hash` `const` in both classes, so it can be called from parallel
   loops. (`Minhash::hash_single` is already `const`.)
4. `hash_single`: bind the `TabulationHash` by `const&` instead of copying the
   8 KB table on every call.
5. Fix `MinhashBuilder::describe()`, which returns `"MinHash1Bit"`, so that it
   returns `"MinHash"`.
6. Add Catch2 tests in `test/include/lsh/hash_test.hpp`, next to the existing
   E2LSH ball-sampler tests:
   - deterministic output under a fixed seed;
   - every `Minhash1Bit` bit is set correctly, including the first repetition;
   - the empirical collision rate on random set pairs is close to
     `collision_probability(jaccard_distance)`.

### Phase 1: C++ value accessors

The value types keep their contents private. Add read-only accessors to
`values.hpp` so the binding can copy codes out without `reinterpret_cast`:

- `BitwiseLshValue<K>::get_bits() const -> uint32_t`. Concatenation `i` sits
  at bit `K-1-i`, consistent with `prefix_eq` and `overwrite`.
- `SymbolLshValue<Symbol, K>::get(size_t i) const -> Symbol` and
  `data() const -> const Symbol*`.

Add a Catch2 check that `get_bits` and `get` agree with `prefix_eq` on random
values.

Also add `float get_quantization_width() const` to `E2LSH` in
`euclidean.hpp`. The field is private, and only `E2LSHBallSampler` can reach
it today, through `friend`.

### Phase 2: binding infrastructure in `python/wrapper.cpp`

1. **`K` dispatch helper.** Add a `template <uint8_t... Ks> struct KList` and a
   `dispatch_k<KList<...>>(k, lambda)` that calls `lambda(std::integral_constant<uint8_t, K>{})`
   for the matching `K` and throws `nb::value_error` otherwise. Each family
   wrapper holds a `std::variant` over its instantiations, following the
   pattern of `EMST_exposed`.
2. **Seeding helper.** Add `maybe_seed(std::optional<uint64_t> seed)`, which
   calls `panna::seed_global_rng` when a seed is set. Document in the
   docstrings that this resets the module-wide RNG.
3. **Dataset conversion.** Add `template <typename Dataset> Dataset to_dataset(const nb::ndarray<float, nb::c_contig>&)`,
   which accepts 1-D or 2-D input, validates `shape(-1) == dimensions`, and
   factors out the copy loop that is currently repeated several times in
   `wrapper.cpp`. Let nanobind convert dtype and contiguity implicitly.
4. **Sparse set conversion.** Add `SparseSets from_csr(indptr, indices)`.
   Validate that `indptr` is monotonic and that `indptr[-1] == len(indices)`.
5. **Output allocation.** Add `make_owned_array<T>(shape)`, which returns a
   `(T*, nb::ndarray)` pair owned by a capsule. Replace the repeated
   `new[]`/capsule boilerplate in the new code.
6. **Parallel hashing kernel.**
   ```cpp
   template <typename Hasher, typename Dataset, typename Writer>
   void hash_rows(Hasher& h, const Dataset& ds, Writer write_row);
   ```
   - Validate inputs and allocate outputs while holding the GIL. Then use
     `nb::gil_scoped_release` and run `#pragma omp parallel for` over the rows,
     with a thread-private `std::vector<Value>`.
   - `CrossPolytope` and `LatticeLSH` index their per-thread scratch by
     `omp_get_thread_num()`. Their scratch is sized at construction with
     `omp_get_max_threads()`, so the kernel must not use more threads than
     that. Pass `num_threads(omp_get_max_threads())` explicitly. A future
     change to `OMP_NUM_THREADS` after construction must not overflow the
     scratch buffers; add a regression test for this.
   - Nothing inside the parallel region may throw. All checks happen before it.
7. **Writers.**
   - Codes, bitwise: unpack `get_bits()` into `K` `uint8` values.
   - Codes, symbolic: copy `K` symbols.
   - Packed, bitwise: write `get_bits()` as `uint32`.
   - Packed, symbolic: `memcpy` the `K*sizeof(Symbol)` bytes into a
     `uint8 (n, reps, nbytes)` buffer. Then, in C++, call
     `.view(numpy.dtype("V{nbytes}"))` and reshape to `(n, reps)`, so that
     Python receives the void dtype directly.

     Caveat for the docs: void keys support equality, `np.unique`, and use as
     dict keys via `.tobytes()`. Their ordering does **not** match the
     library's `prefix_less` ordering, because the symbols are little-endian
     and signed.
8. **`collision_probability`.** Overload it on `float` and on
   `nb::ndarray<float>`. The array version loops over the elements, which is
   cheap enough without the GIL release.

### Phase 3: family classes

For each family, add a wrapper struct (`PySimhash`, `PyCrossPolytope`, ...)
that holds the variant and the metadata (`k`, `repetitions`, `dimensions`,
fitted parameters). Register it in `NB_MODULE` with `nb::class_`, named
keyword arguments with defaults (`nb::arg("k") = ...`), and docstrings.

Default `k` values follow what the library already uses:

| Family | Default `k` |
|---|---|
| Simhash | 16 (24, the value used in the EMST/TrieIndex wrapper, is not a power of two) |
| CrossPolytope | 4 |
| E2LSH | 8 |
| LatticeLSH | 4 |
| MinHash | 4 |
| MinHash1Bit | 16 |

`repetitions` has no default and must be passed.

Notes per family:

- **Simhash / CrossPolytope.** Built through their builders, which have no
  data-dependent fit. CrossPolytope also exposes `estimation_repetitions` and
  `estimation_eps`, which control how its collision-probability table is
  estimated.
- **E2LSH.**
  - Constructor: `E2LSH(quantization_width, dimensions, repetitions)`.
  - `fit`: `E2LSHBuilder<K, EuclideanPoints, EuclideanDistance>(dimensions).fit(points, repetitions, delta)`,
    then `build(repetitions)`. The fit runs with the GIL released.
  - Expose the fitted `quantization_width` through the new
    `E2LSH::get_quantization_width()` (Phase 1).
  - Document that codes are `int8` and can wrap around when the quantization
    width is very small relative to the data range.
- **LatticeLSH.**
  - Constructor: `LatticeLSH(offset_vec, scaling_factor, dimensions, repetitions)`.
  - `fit`: `LatticeLSHBuilder<K>(dimensions).fit(points, repetitions, delta)`.
    (Corrected during implementation: the 4-argument overload does use the
    distance upper bound, and passing 0 yields a tiny scaling factor; the
    3-argument overload is the data-driven one.) Read back the fitted `offset` and `scaling_factor`; this
    needs getters on `LatticeLSHBuilder` (small C++ addition).
  - Document that the `PANNA_LATTICE_SCALING_FACTOR` environment variable
    still overrides fitting.
- **MinHash / MinHash1Bit.**
  - No `dimensions` property.
  - `collision_probability` takes a Jaccard distance.
  - `hash_sets` is implemented in Python as a small module-level shim added
    to `python/pypanna/__init__.py`, attached to the classes, because it
    detects scipy CSR matrices (`.indptr` / `.indices`) and lists. scipy is
    imported lazily; it is already a transitive dependency via scikit-learn.
    Document that the empty set hashes to token 0.

### Phase 4: type stubs

1. In `CMakeLists.txt`, add `nanobind_add_stub(_panna_impl_stub MODULE _panna_impl OUTPUT _panna_impl.pyi PYTHON_PATH $<TARGET_FILE_DIR:_panna_impl> DEPENDS _panna_impl)`
   and install the `.pyi` next to the extension in the `pypanna` package.
2. Add an empty `python/pypanna/py.typed`.
3. Use `nb::sig` where nanobind's generated signatures are unhelpful, for
   example the `packed` return types and the static `fit`.
4. Check that the stub is present in the wheel built by
   `pip install . --no-build-isolation`.

### Phase 5: pytest tests (`python/tests/test_lsh.py`)

Add `pytest` to a `[dependency-groups] dev` entry in `pyproject.toml`.

For every family, using small random data:

- **Shapes and dtypes:** `hash` returns `(n, reps, k)` with the documented
  dtype; `packed=True` returns `(n, reps)` with `uint32` or `V{nbytes}`; 1-D
  input works.
- **Unsupported `k`:** raises `ValueError` and the message lists the allowed
  values.
- **Determinism:** the same seed gives identical hashers and outputs; a
  different seed gives different outputs.
- **Packed vs codes consistency:** two rows have equal packed keys if and only
  if their codes are equal in all `k` positions.
- **Parallel vs sequential:** hashing all rows at once equals hashing them one
  by one. This guards the per-thread scratch logic.
- **Collision probability:** for pairs at controlled distances, the empirical
  per-repetition collision rate with `k=1` is within a binomial tolerance of
  `collision_probability(d)`.
  - Simhash: pairs at a chosen angle, converted to cosine distance
    (`collision_probability` takes cosine distance, like CrossPolytope).
  - CrossPolytope: cosine distance.
  - E2LSH and LatticeLSH: Euclidean distance with an explicit
    width or scaling factor.
  - MinHash and MinHash1Bit: Jaccard distance of constructed sets.
- **Fit (E2LSH, LatticeLSH):** `fit` returns a hasher with a positive
  parameter, and empty data raises `ValueError`.
- **Validation:** a dimension mismatch, a bad `offset` length, and malformed
  CSR input all raise `ValueError`.
- **Cosine normalization:** scaling an input row does not change its
  Simhash/CrossPolytope hash.

## Files touched

- `include/panna/lsh/minhash.hpp`: Phase 0 fixes
- `include/panna/lsh/values.hpp`: accessors
- `include/panna/lsh/euclidean.hpp`: `E2LSH::get_quantization_width()`
- `include/panna/lsh/lattice.hpp`: `LatticeLSHBuilder` getters
- `test/include/lsh/hash_test.hpp`: C++ tests
- `python/wrapper.cpp`: helpers and six family classes
- `python/pypanna/__init__.py`: the `hash_sets` shim
- `python/pypanna/py.typed`: new
- `python/tests/test_lsh.py`: new
- `CMakeLists.txt`: stub generation and install
- `pyproject.toml`: pytest dev dependency

## Risks and open points

- **Compile time and binary size.** 32 extra instantiations, plus the
  CrossPolytope estimate tables computed at construction. Measure the wheel
  size before and after. If it grows too much, drop `K=1` or cap the symbolic
  families at 8.
- **Global-RNG seeding.** Seeding through the global RNG is not thread-safe
  across concurrent Python threads constructing hashers. Document this. A
  per-object RNG, as `LatticeLSH` already supports, is a possible follow-up.
- **Per-thread scratch.** The scratch-buffer assumption in `CrossPolytope`
  and `LatticeLSH` (`omp_get_thread_num()` < the thread count at construction)
  is fragile. Phase 2 guards it explicitly and Phase 5 tests it.

## Implementation summary (2026-11-05)

All phases are implemented. Nothing is committed yet.

### Verification

- `pytest` (from the repo root, inside `nix develop`, after
  `pip install --no-build-isolation .`): 127 passed in about 12 s. It also
  passes three times in a row and with `OMP_NUM_THREADS=1`. The collision
  tests were run with 20 seeds, and the worst deviation used 48% of the
  5σ tolerance.
- C++ `build/cpp/tests "[lsh]"`: all pass (8 test cases).
- The full C++ suite: 84/86 pass. The 2 failures (`pseudorandom rotations`,
  and the order-dependent `EMST mutual reachability distance`) fail the same
  way without these changes.
- mypy and pyright on an example script that uses the stubs: they report
  exactly the 3 errors planted in it.
- Extension size: 2,374,896 bytes, up from 1,850,576 (+28%). A full build
  takes about 60–80 s.

### Changes by file

- `include/panna/lsh/minhash.hpp`: the Phase 0 fixes as planned.
- `include/panna/lsh/values.hpp`: `BitwiseLshValue::get_bits()`, and
  `SymbolLshValue::get(i)` and `data()`.
- `include/panna/lsh/euclidean.hpp`: `E2LSH::get_quantization_width()`.
- `include/panna/lsh/lattice.hpp`: `get_offset()` and `get_scaling_factor()`
  on both `LatticeLSH` and `LatticeLSHBuilder`.
- `include/panna/data.hpp` (not planned): `SparseSets::operator[]` no longer
  throws for an empty set stored last.
- `include/panna/rand.hpp` (not planned):
  - `seed_global_rng` now resets the shared normal distribution. Before, a
    cached spare sample meant the same seed could give different hashers.
  - `sample_int` no longer uses a `static` distribution, which kept the
    bounds of its first call. This broke `LatticeLSH.fit` on a large dataset
    followed by a smaller one.
- `python/wrapper.cpp`:
  - The helpers from Phase 2 and the six family classes.
  - A module-level `global_rng_mutex`. Hasher construction, `fit` and
    `set_seed` release the GIL and seed/sample under this mutex.
  - The five old copy loops now use `to_dataset`.
- `python/pypanna/__init__.py`: the `hash_sets` shim.
- `python/pypanna/py.typed` and `python/stubgen_patterns.txt` (new). The
  pattern file adds `hash_sets` to the generated stub.
- `CMakeLists.txt`: `nanobind_add_stub` and installation of the `.pyi`.
- `pyproject.toml`: a `dev` dependency group with pytest, and
  `[tool.pytest.ini_options]` (`testpaths`, `--import-mode=importlib`).
  `uv.lock` was updated offline.
- Tests:
  - `test/include/lsh/hash_test.hpp`: MinHash tests.
  - `test/include/lsh/values_test.hpp`: accessor tests against
    `prefix_eq`.
  - `test/include/lsh/lattice_test.hpp`: getter tests.
  - `python/tests/test_lsh.py` (new).

### Deviations from the plan

- **Signature order.** Constructors take `dimensions, repetitions`, then the
  required family parameter, then `k` and the other defaults. All arguments
  work as keywords.
- **`collision_probability`.**
  - It returns the probability for a single concatenation, not `**k`.
  - The cosine families take a cosine distance.
  - NumPy scalars return a Python `float`. Any ndarray, including size 1 or
    0-d, returns a `float32` array of the same shape.
- **Thread safety.** `hash` on `CrossPolytope` and `LatticeLSH`, whose
  `hash` is not `const`, is serialized per object with a mutex. Without it,
  concurrent Python threads would share scratch slots. The OpenMP team is
  capped at `min(threads at construction, current max threads)`.
- **`fit`.**
  - It takes 2-D data only.
  - `LatticeLSH.fit` needs at least 2 points; the EMST code is undefined on
    1 point.
  - `E2LSH.fit` raises if the fitted width is not positive (e.g. all-zero
    data).
  - `E2LSH.fit` accepts `delta` and validates it, but the C++ heuristic
    ignores it.
- **`from_csr`.** It also requires `indptr[0] == 0`. Tokens come in as
  int64, so negative values are rejected instead of wrapping.
- **CrossPolytope `dimensions`.** It is limited to 16384, so codes fit in
  `int16`.
- **Stubs.**
  - `hash` uses `@overload` on `packed: Literal[False|True]`.
  - Dense inputs are typed `NDArray[Any]`, not `ArrayLike`, because Python
    lists are rejected at runtime.
- **Not tested.** Stub generation in an isolated build (cibuildwheel). It
  is expected to work, because stubgen does not need NumPy.

### Follow-ups noticed (not done)

- `LatticeLSHBuilder(float offset, float scaling_factor, size_t dims)`
  initializes `std::vector<float> offset(offset)` with the float as a
  *size*. This is a bug, and the bindings avoid this constructor.
- `SparseSets::push_back` rejects `token > dimensions`; it should probably
  be `>=`.
- The existing `MinHash1Bit` collision test in `hash_test.hpp` uses
  `accepted_deviation = 10`, so it can never fail.
- `sample_random_normal(std::mt19937_64&)` still keeps a static distribution
  with a cached spare sample. It is only used by `E2LSHBallSampler`.
- `EMST`/`TrieIndex` use the global RNG without `global_rng_mutex`.
- With the default `estimation_eps=5e-3`, `CrossPolytope` construction takes
  several seconds per instance, because it estimates the probability table.
