"""Exact k-nearest-neighbor graph construction."""

import numpy as np
from numba import njit, prange, get_num_threads
from pathlib import Path

__all__ = ["exact_knn_graph"]


@njit(parallel=True, fastmath=True, cache=True)
def _exact_knn_kernel(data, k, block_size, first_block, last_block, neighbors, distances):
    n, d = data.shape

    # Blocks are disjoint, hence the output does not depend on how many
    # threads numba decides to use.
    for b in prange(first_block, last_block):
        begin = b * block_size
        end = min(begin + block_size, n)
        rows = end - begin

        # Top-k candidates of each point of the block, kept sorted by
        # increasing (squared) distance. Empty slots hold +infinity.
        best_dist = np.full((rows, k), np.inf, dtype=np.float32)
        best_idx = np.zeros((rows, k), dtype=np.uint32)

        # Each candidate row is loaded once per block, and reused by all the
        # points of the block.
        for j in range(n):
            for r in range(rows):
                i = begin + r
                if i == j:
                    continue
                # Keep this loop free of branches so that it can be vectorized.
                dist = np.float32(0.0)
                for t in range(d):
                    diff = data[i, t] - data[j, t]
                    dist += diff * diff
                if dist < best_dist[r, k - 1]:
                    p = k - 1
                    while p > 0 and best_dist[r, p - 1] > dist:
                        best_dist[r, p] = best_dist[r, p - 1]
                        best_idx[r, p] = best_idx[r, p - 1]
                        p -= 1
                    best_dist[r, p] = dist
                    best_idx[r, p] = j

        for r in range(rows):
            i = begin + r
            for c in range(k):
                distances[i, c] = np.sqrt(best_dist[r, c])
                neighbors[i, c] = best_idx[r, c]


def exact_knn_graph(
    data: np.ndarray, k: int, block_size: int = 64, progress: bool = True
):
    """Compute the exact k-nearest neighbor graph of `data` by brute force.

    Returns the pair `(neighbors, distances)`, both of shape `(n, k)`, where row
    `i` lists the k points closest to point `i` in order of increasing Euclidean
    distance. A point is never its own neighbor. Distances are actual Euclidean
    distances, not squared ones, computed in single precision to match the C++
    side of panna. Ties between equally distant points are broken arbitrarily.

    The computation is O(n^2 * d) and runs in parallel over blocks of `block_size`
    points; the number of threads is the one numba is configured with, see the
    NUMBA_NUM_THREADS environment variable.

    Unless `progress` is False, a tqdm progress bar reports how many points have
    been processed so far.
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-dimensional array, got {data.ndim} dimensions")
    n = data.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k should be between 1 and {n - 1} (n={n}), got {k}")
    if block_size < 1:
        raise ValueError(f"block_size should be at least 1, got {block_size}")

    neighbors = np.empty((n, k), dtype=np.uint32)
    distances = np.empty((n, k), dtype=np.float32)

    num_blocks = (n + block_size - 1) // block_size
    # Each call to the kernel handles enough blocks to keep all the threads
    # busy; the progress bar is updated once per call.
    blocks_per_chunk = max(1, get_num_threads())

    if progress:
        from tqdm import tqdm

        bar = tqdm(total=n, unit="points", desc="knn")
    else:
        bar = None

    try:
        for first_block in range(0, num_blocks, blocks_per_chunk):
            last_block = min(first_block + blocks_per_chunk, num_blocks)
            _exact_knn_kernel(
                data, k, block_size, first_block, last_block, neighbors, distances
            )
            if bar is not None:
                done = min(last_block * block_size, n)
                bar.update(done - bar.n)
    finally:
        if bar is not None:
            bar.close()

    return neighbors, distances


def write_knn(data: np.ndarray, path: Path, k: int, block_size: int = 64):
    """Compute the knn graph of the given data, and writes it in the given hdf5 file."""
    import h5py
    # warmup run to ensure the kernel is compiled
    _= exact_knn_graph(data[:100], 10, block_size, progress=False)

    # the actual run
    neighbors, distances = exact_knn_graph(data, k, block_size)
    with h5py.File(path, "a") as hfp:
        if "/graph/neighbors" in hfp:
            raise ValueError("File already contains a graph")
        hfp["/graph/neighbors"] = neighbors
        hfp["/graph/distances"] = distances


if __name__ == "__main__":
    import argparse

    try:
        from . import datasets
    except ImportError:
        # Allow running this file directly, without the compiled extension
        # that `panna/__init__.py` pulls in.
        import datasets

    parser = argparse.ArgumentParser(
        description="Compute the exact knn graph of a dataset and store it in a hdf5 file."
    )
    parser.add_argument("k", type=int, help="number of neighbors of each point")
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
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="number of points handled by each parallel block",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = datasets.local_path(args.dataset)

    _, data = datasets.load(args.dataset, normalize="angular" in dataset)
    write_knn(data, output, args.k, args.block_size)
    print(f"wrote the {args.k}-nn graph of {args.dataset} to {output}")
