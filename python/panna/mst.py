"""Exact minimum spanning tree under the mutual reachability distance."""

import numpy as np
from numba import njit

__all__ = ["exact_mst"]

# Layout of the small state vector threaded through `_prim_kernel`, which is
# resumable so that the caller can report progress between chunks of edges.
_STARTED = 0
_N_REMAINING = 1
_HEAP_SIZE = 2
_N_OUTSIDE = 3
_NUM_EDGES = 4
_STATE_SIZE = 5


@njit(cache=True, inline="always")
def _heap_less(hkey, htgt, hsrc, a, b):
    # The full (key, target, source) triple is compared, so that ties in the
    # weights are broken on the vertex indices and the tree does not depend on
    # the order in which equally-weighted edges happen to be pushed.
    if hkey[a] != hkey[b]:
        return hkey[a] < hkey[b]
    if htgt[a] != htgt[b]:
        return htgt[a] < htgt[b]
    return hsrc[a] < hsrc[b]


@njit(cache=True)
def _heap_push(hkey, htgt, hsrc, size, key, tgt, src):
    i = size
    hkey[i] = key
    htgt[i] = tgt
    hsrc[i] = src
    while i > 0:
        p = (i - 1) // 2
        if not _heap_less(hkey, htgt, hsrc, i, p):
            break
        hkey[i], hkey[p] = hkey[p], hkey[i]
        htgt[i], htgt[p] = htgt[p], htgt[i]
        hsrc[i], hsrc[p] = hsrc[p], hsrc[i]
        i = p
    return size + 1


@njit(cache=True)
def _heap_pop(hkey, htgt, hsrc, size):
    """Remove the root. The caller is expected to have read it beforehand."""
    size -= 1
    hkey[0] = hkey[size]
    htgt[0] = htgt[size]
    hsrc[0] = hsrc[size]
    i = 0
    while True:
        left = 2 * i + 1
        right = left + 1
        smallest = i
        if left < size and _heap_less(hkey, htgt, hsrc, left, smallest):
            smallest = left
        if right < size and _heap_less(hkey, htgt, hsrc, right, smallest):
            smallest = right
        if smallest == i:
            break
        hkey[i], hkey[smallest] = hkey[smallest], hkey[i]
        htgt[i], htgt[smallest] = htgt[smallest], htgt[i]
        hsrc[i], hsrc[smallest] = hsrc[smallest], hsrc[i]
        i = smallest
    return size


@njit(cache=True)
def _closest_outside(u, data, cand_id, cand_dist, cand_bound, core, in_tree,
                     remaining, n_remaining):
    """Node outside the tree at minimum mutual reachability distance from `u`.

    Returns the pair `(weight, node)`, or `(inf, -1)` when the tree already
    spans every point.

    Row `u` of `cand_id`/`cand_dist` is a cache of points close to `u`, sorted
    by increasing Euclidean distance and padded with `-1`, and `cand_bound[u]`
    is a lower bound on the distance from `u` to any point not in the row.
    The cache starts out as the k-nearest neighbor graph and is refilled by
    brute force whenever it can no longer answer the query.
    """
    c = cand_id.shape[1]
    cu = core[u]
    best = np.float32(np.inf)
    arg = np.int64(-1)

    # First look in the cache: if the answer is there, this costs O(c) instead
    # of a scan of the whole dataset. Note that the row cannot be cut short at
    # its first unconnected entry when the core distances are not all zero: it
    # is sorted by Euclidean distance, which is not the order induced by the
    # mutual reachability distance.
    for j in range(c):
        v = np.int64(cand_id[u, j])
        if v < 0 or in_tree[v]:
            continue
        w = cand_dist[u, j]
        if cu > w:
            w = cu
        if core[v] > w:
            w = core[v]
        if arg < 0 or w < best or (w == best and v < arg):
            best = w
            arg = v

    # The cache is only a partial view of the graph, so the candidate found
    # above has to be certified. Every point `x` outside the row is at
    # Euclidean distance at least `cand_bound[u]` from `u`, hence its mutual
    # reachability distance from `u` is at least `max(cand_bound[u], cu)`:
    # anything not exceeding that bound is a global minimum.
    bound = cand_bound[u]
    if cu > bound:
        bound = cu
    if arg >= 0 and best <= bound:
        return best, arg

    # Otherwise scan by brute force the points still outside the tree, taking
    # the minimum mutual reachability distance among all of them. `remaining`
    # holds a superset of those points, kept compact by the caller, so this is
    # proportional to the number of points still outside the tree rather than
    # to n.
    #
    # The same scan refills the cache with the `c` points closest to `u` in
    # Euclidean terms, which is what makes the bound above usable: points
    # already connected are left out on purpose, since they will never become
    # candidates again. The answer itself cannot be read off those `c` points,
    # though: a farther point with a small core distance beats a nearer one
    # with a large one, which is why the minimum is tracked separately.
    dims = data.shape[1]
    best = np.float32(np.inf)
    arg = np.int64(-1)
    for j in range(c):
        cand_id[u, j] = -1
        cand_dist[u, j] = np.inf
    # Squared distance of the farthest entry currently cached, so that the
    # insertion is skipped for most of the points.
    threshold = np.float32(np.inf)
    for t in range(n_remaining):
        v = np.int64(remaining[t])
        if in_tree[v]:
            continue
        acc = np.float32(0.0)
        for x in range(dims):
            diff = data[u, x] - data[v, x]
            acc += diff * diff
        dist = np.sqrt(acc)
        w = dist
        if cu > w:
            w = cu
        if core[v] > w:
            w = core[v]
        if arg < 0 or w < best or (w == best and v < arg):
            best = w
            arg = v
        if acc < threshold:
            p = c - 1
            while p > 0 and cand_dist[u, p - 1] > dist:
                cand_dist[u, p] = cand_dist[u, p - 1]
                cand_id[u, p] = cand_id[u, p - 1]
                p -= 1
            cand_dist[u, p] = dist
            cand_id[u, p] = v
            threshold = cand_dist[u, c - 1] * cand_dist[u, c - 1]
    cand_bound[u] = cand_dist[u, c - 1]
    return best, arg


@njit(cache=True)
def _prim_kernel(
    data,
    cand_id,
    cand_dist,
    cand_bound,
    core,
    in_tree,
    remaining,
    hkey,
    htgt,
    hsrc,
    out_src,
    out_dst,
    out_weight,
    state,
    target_edges,
):
    """Run Prim's algorithm until `target_edges` edges have been emitted.

    The priority queue is keyed by the vertices *inside* the tree: each of them
    holds a single entry `(weight, target, source)`, where `target` is its
    closest vertex outside the tree. The minimum of those entries is the
    lightest edge crossing the cut, which is exactly what Prim's algorithm
    needs. Absorbing a vertex into the tree can only move a key up, never down,
    so the stale entries left behind are lower bounds and lazy deletion is
    sound: popping a stale entry recomputes it and pushes it back without
    emitting an edge.
    """
    n = data.shape[0]
    n_remaining = state[_N_REMAINING]
    heap_size = state[_HEAP_SIZE]
    n_outside = state[_N_OUTSIDE]
    num_edges = state[_NUM_EDGES]

    if state[_STARTED] == 0:
        state[_STARTED] = 1
        in_tree[0] = True
        n_remaining = n
        n_outside = n - 1
        w, v = _closest_outside(
            0, data, cand_id, cand_dist, cand_bound, core, in_tree, remaining,
            n_remaining
        )
        if v >= 0:
            heap_size = _heap_push(hkey, htgt, hsrc, heap_size, w, v, 0)

    while num_edges < target_edges:
        if heap_size == 0:
            # Cannot happen: every vertex of the tree keeps an entry until the
            # tree spans the whole dataset. Guard against it anyway, since the
            # kernel indexes arrays without bounds checks.
            raise RuntimeError(
                "the priority queue emptied before the tree was complete"
            )
        key = hkey[0]
        tgt = np.int64(htgt[0])
        src = np.int64(hsrc[0])
        heap_size = _heap_pop(hkey, htgt, hsrc, heap_size)

        if in_tree[tgt]:
            w, v = _closest_outside(
                src, data, cand_id, cand_dist, cand_bound, core, in_tree,
                remaining, n_remaining
            )
            if v >= 0:
                heap_size = _heap_push(hkey, htgt, hsrc, heap_size, w, v, src)
            continue

        out_src[num_edges] = src
        out_dst[num_edges] = tgt
        out_weight[num_edges] = key
        num_edges += 1
        in_tree[tgt] = True
        n_outside -= 1

        # Drop the absorbed vertices from `remaining` once they are more
        # numerous than the ones still outside, keeping the brute force scans
        # proportional to `n_outside`. The relative order is preserved, so the
        # scans stay deterministic.
        if n_remaining > 2 * n_outside + 64:
            p = 0
            for t in range(n_remaining):
                v = remaining[t]
                if not in_tree[v]:
                    remaining[p] = v
                    p += 1
            n_remaining = p

        w, v = _closest_outside(
            src, data, cand_id, cand_dist, cand_bound, core, in_tree,
            remaining, n_remaining
        )
        if v >= 0:
            heap_size = _heap_push(hkey, htgt, hsrc, heap_size, w, v, src)
        w, v = _closest_outside(
            tgt, data, cand_id, cand_dist, cand_bound, core, in_tree,
            remaining, n_remaining
        )
        if v >= 0:
            heap_size = _heap_push(hkey, htgt, hsrc, heap_size, w, v, tgt)

    state[_N_REMAINING] = n_remaining
    state[_HEAP_SIZE] = heap_size
    state[_N_OUTSIDE] = n_outside
    state[_NUM_EDGES] = num_edges


def exact_mst(
    data: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    minPts: int = 1,
    progress: bool = True,
):
    """Compute the exact minimum spanning tree of `data` under the mutual
    reachability distance.

    `neighbors` and `distances` are the exact k-nearest neighbor graph of
    `data`, as returned by `panna.exact_knn_graph`: row `i` of `neighbors`
    lists the k points closest to point `i`, excluding `i` itself, and row `i`
    of `distances` holds the corresponding Euclidean distances. `minPts` is the
    number of points defining the core distances, so `minPts == 1` gives the
    plain Euclidean minimum spanning tree; `neighbors` must have at least
    `minPts` columns.

    The mutual reachability distance between two points is
    `max(d(x, y), core(x), core(y))`, where `core(x)` is the distance from `x`
    to its `minPts`-th nearest neighbor (zero when `minPts == 1`).

    Returns the same triple as
    `fast_hdbscan.hdbscan.compute_minimum_spanning_tree`:

    - `mst_edges`, of shape `(n - 1, 3)` and dtype float64, whose columns are
      `[src, dst, weight]`; the weights are mutual reachability distances, and
      the rows are sorted by increasing weight;
    - `neighbors`, the input neighbor graph as int32;
    - `core_distances`, of shape `(n,)` and dtype float32.

    The tree is exact, but ties between equally weighted edges may be resolved
    differently than in other implementations, in which case the tree differs
    while its weight does not.

    The neighbor graph is only used to answer, cheaply, the queries Prim's
    algorithm asks; whenever it is too coarse to do so with certainty, the
    query falls back to a scan of the points not yet connected. The cost is
    therefore `O(n * k)` plus those scans, degrading in the worst case to the
    `O(n^2 * d)` of a dense Prim. Two arrays of the size of the neighbor graph
    are allocated to cache the results of the scans.

    Unless `progress` is False, a tqdm progress bar reports how many edges have
    been found so far.
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-dimensional array, got {data.ndim} dimensions")
    n = data.shape[0]

    neighbors = np.ascontiguousarray(neighbors)
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    if neighbors.ndim != 2 or distances.ndim != 2:
        raise ValueError("neighbors and distances should be 2-dimensional arrays")
    if neighbors.shape != distances.shape:
        raise ValueError(
            f"neighbors and distances should have the same shape, got "
            f"{neighbors.shape} and {distances.shape}"
        )
    if neighbors.shape[0] != n:
        raise ValueError(
            f"neighbors should have one row per point, got {neighbors.shape[0]} "
            f"rows for {n} points"
        )
    k = neighbors.shape[1]
    if minPts < 1:
        raise ValueError(f"minPts should be at least 1, got {minPts}")
    if k < minPts:
        raise ValueError(
            f"the neighbor graph should have at least minPts={minPts} columns, got {k}"
        )
    neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
    if n > 0 and (neighbors.min() < 0 or neighbors.max() >= n):
        raise ValueError("neighbors contains out of range point ids")

    # Same convention as fast_hdbscan: with minPts == 1 the core distances are
    # zero, so the mutual reachability distance collapses to the Euclidean one.
    if minPts == 1:
        core = np.zeros(n, dtype=np.float32)
    else:
        core = np.ascontiguousarray(distances[:, minPts - 1], dtype=np.float32)

    if n <= 1:
        return np.empty((0, 3), dtype=np.float64), neighbors, core

    # The candidate cache starts out as the neighbor graph and is then
    # refilled in place, so it has to be a copy: `neighbors` is handed back to
    # the caller untouched.
    cand_id = neighbors.copy()
    cand_dist = distances.copy()
    cand_bound = distances.max(axis=1)

    in_tree = np.zeros(n, dtype=np.bool_)
    remaining = np.arange(n, dtype=np.int32)
    # Each vertex of the tree owns exactly one heap entry, which is popped and
    # pushed back on every visit, so the heap never holds more than n entries.
    hkey = np.empty(n, dtype=np.float32)
    htgt = np.empty(n, dtype=np.int32)
    hsrc = np.empty(n, dtype=np.int32)
    out_src = np.empty(n - 1, dtype=np.int64)
    out_dst = np.empty(n - 1, dtype=np.int64)
    out_weight = np.empty(n - 1, dtype=np.float32)
    state = np.zeros(_STATE_SIZE, dtype=np.int64)

    # The kernel is resumable: it is called on chunks of edges so that the
    # progress bar can be updated in between.
    chunk = max(1, (n - 1) // 100) if progress else n - 1

    if progress:
        from tqdm import tqdm

        bar = tqdm(total=n - 1, unit="edges", desc="mst")
    else:
        bar = None

    try:
        while state[_NUM_EDGES] < n - 1:
            target = min(state[_NUM_EDGES] + chunk, n - 1)
            _prim_kernel(
                data,
                cand_id,
                cand_dist,
                cand_bound,
                core,
                in_tree,
                remaining,
                hkey,
                htgt,
                hsrc,
                out_src,
                out_dst,
                out_weight,
                state,
                target,
            )
            if bar is not None:
                bar.update(int(state[_NUM_EDGES]) - bar.n)
    finally:
        if bar is not None:
            bar.close()

    order = np.argsort(out_weight, kind="stable")
    edges = np.empty((n - 1, 3), dtype=np.float64)
    edges[:, 0] = out_src[order]
    edges[:, 1] = out_dst[order]
    edges[:, 2] = out_weight[order]
    return edges, neighbors, core

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import h5py

    from panna import datasets
    from panna import knn

    parser = argparse.ArgumentParser(
        description="Compute the exact emst of a dataset under the mutual reachability distance and store it in a hdf5 file."
    )
    parser.add_argument("minPts", type=int, help="number of neighbors in the clustering")
    parser.add_argument(
        "dataset",
        choices=datasets.available_datasets(),
        metavar="dataset",
        help="name of the dataset to load, one of "
        + ", ".join(datasets.available_datasets()),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output hdf5 file (default: dataset's local file)",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = datasets.local_path(args.dataset)

    _, data = datasets.load(args.dataset, normalize="angular" in dataset)
    with h5py.File(output) as hfp:
        if "/graph" not in hfp:
            print("computing the knn graph")
            knn.write_knn(data, output, args.minPts)
        distances = hfp["/graph/distances"][:]
        neighbors = hfp["/graph/neighbors"][:]

    minPts = args.minPts
    tree, neighbors, core = exact_mst(
        data, neighbors, distances, minPts, progress=True
    )
    with h5py.File(output, "a") as hfp:
        hfp[f"/tree-{minPts}/tree"] = tree
        hfp[f"/tree-{minPts}/neighbors"] = neighbors
        hfp[f"/tree-{minPts}/core"] = core
    
    print(f"wrote the EMST under the {minPts}-reachability distance of {args.dataset} to {output}")
