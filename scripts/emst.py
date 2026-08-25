#!/usr/bin/env python
"""
This script runs all the experiments regarding the EMST, including the baselines
"""

from pandas.core.frame import infer_dtype_from_object

import panna
import dataclasses
import polars as pl
from dataclasses import dataclass, asdict
from icecream import ic
from pathlib import Path
import platform
import hashlib
import numpy as np
import time
import json
import tempfile
import math
from filelock import FileLock
from datetime import datetime
import argparse
import multiprocessing
import resource
import fast_hdbscan
import gzip


# We do not use an actual database, but store results in a newline-delimited json file,
# because then it's more friendly to store it in git along with the code to keep track
# of the history of the experiments, facilitating merges
DATABASE_DIR = Path("results")
DATABASE_FILE = DATABASE_DIR / "emst.json"
LOCKFILE = DATABASE_DIR / "emst.lock"
TIMEOUT_S = 2 * 3600


def get_git_version():
    import subprocess

    if hasattr(panna, "git_version"):
        return panna.git_version
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        return ""


def get_version(algorithm: str):
    from importlib.metadata import version

    if algorithm in ("k+", "k+scan"):
        return dict(version=panna.EMST.version, git_version=get_git_version())
    elif algorithm == "tutte":
        return dict(version=version("fast_hdbscan"), git_version="")
    elif algorithm == "mlpack":
        return dict(version=version("mlpack"), git_version="")
    elif algorithm == "pyhdbscan":
        return dict(version=version("pyhdbscan"), git_version="")
    elif algorithm == "hssl":
        return dict(version=version("hnswhsslrust"), git_version="")
    else:
        raise ValueError(f"unknown algorithm `{algorithm}`")


def get_processor_name():
    # Source - https://stackoverflow.com/a/13078519
    # Posted by dbn, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-03-02, License - CC BY-SA 4.0
    import os
    import platform
    import subprocess
    import re

    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1).strip()
    return ""


def get_machine_info() -> dict:
    nodename = platform.node()
    if "lovelace" in nodename:
        # Consider all nodes of the lovelace cluster the same,
        # for the purpose of building a primary key
        nodename = "lovelace"
    return {
        "processor": get_processor_name(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "system": platform.system(),
        "node_name": nodename,
    }


def get_commit_date(git_version: str) -> str | None:
    """Retrieves the commit date for a given git commit hash."""
    import subprocess
    if not git_version:
        return None
    try:
        # %ci gives committer date, ISO 8601 format
        # -s suppresses diff output, only shows commit message
        date_str = subprocess.check_output(
            ["git", "show", "-s", "--format=%ci", git_version],
            stderr=subprocess.DEVNULL # Suppress errors for non-existent commits
        ).decode("ascii").strip()
        return date_str
    except subprocess.CalledProcessError:
        # This can happen if the git_version is not a valid commit or git is not available
        return None
    except FileNotFoundError:
        # git command not found
        return None


class HashWriter:
    """Computes the hash of an object as it's being written."""
    def __init__(self):                                                                        
        self.hasher = hashlib.sha256()
    def write(self, data):                                                                             
        self.hasher.update(data)                                                                       
        return len(data)
    def flush(self):
        pass
    def hexdigest(self):
        return self.hasher.hexdigest()

def profile_sha_path(profile_list):
    h = HashWriter()
    profile = pl.DataFrame(profile_list)
    profile.write_parquet(h)
    return str(DATABASE_DIR / (h.hexdigest() + ".pq"))


def data_sha(array: np.ndarray) -> str:
    """return the string representing the sha512 code for the given numpy array"""
    return hashlib.sha512(array.tobytes()).hexdigest()


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def compute_flexibility(tree, epsilon, diameter):
    total_cost = sum(tree)
    cost = 0
    for i, w in enumerate(tree):
        remaining = len(tree) - i
        cost += w
        lower_bound = remaining * w
        upper_bound = remaining * diameter
        if upper_bound <= epsilon * cost:
            return remaining
    return 0


def compute_edge_mass(weights, counts, threshold):
    idx = np.searchsorted(weights, threshold, side="right")
    # a threshold beyond the last bucket boundary lands past the end of the
    # cumulative counts: in that case the whole mass is below the threshold
    return counts[min(idx, len(counts) - 1)]


def estimate_contrast(edge_mass, bounds, cumulative_counts, diameter):
    def find(mass):
        idx = np.searchsorted(cumulative_counts, mass)
        if idx >= len(bounds):
            return diameter
        ic(mass, idx, bounds[idx])
        return bounds[idx]
    return find(2*edge_mass) / find(edge_mass)

def compute_cumulative_distance_distribution(
    data, min_distance, max_distance, num_buckets=10000, sample_fraction=0.01
):
    n = data.shape[0]
    num_pairs = n * (n - 1) // 2
    samples = int(min(1e9, num_pairs * sample_fraction))
    counts, bounds = panna.distance_histogram(
        data, num_buckets, min_distance, max_distance, samples
    )
    mean_weight = np.average(bounds, weights=counts)
    counts = np.cumsum(counts)
    return bounds, counts, mean_weight

@dataclass
class Entry(object):
    version: str
    git_version: str
    algorithm: str
    parameters: dict
    dataset: str
    dataset_sample_frac: float | None
    dataset_sample_seed: int
    dataset_sha: str
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now().isoformat()
    )
    machine: dict = dataclasses.field(default_factory=get_machine_info)
    running_time_s: float | None = None
    memory_kb: int | None = None
    emst_weight: float | None = None
    detail: dict = dataclasses.field(default_factory=dict)
    profile_path: str | None = None

    def as_dict(self):
        return asdict(self)

    def primary_key(self):
        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "machine": self.machine,
            "dataset": self.dataset,
            "dataset_sample_frac": self.dataset_sample_frac,
            "dataset_sample_seed": self.dataset_sample_seed,
            "dataset_sha": self.dataset_sha,
        }


def already_run(key: dict) -> bool:
    """Check if a configuration with the given key has already been run"""
    if not DATABASE_FILE.is_file():
        return False
    with FileLock(LOCKFILE):
        df = pl.read_ndjson(DATABASE_FILE, infer_schema_length=None)
        predicate = [
            pl.col(k).is_null() if v is None else (pl.col(k) == v)
            for k, v in key.items()
        ]
        df = df.filter(
            (pl.col("running_time_s").is_null()) | (pl.col("running_time_s") >= 0)
        )
        return len(df.filter(predicate)) > 0


def tree_weight(data, edges):
    xs = data[edges[:, 0]]
    ys = data[edges[:, 1]]
    ws = np.linalg.norm(xs - ys, axis=1)
    return float(ws.sum()), ws


def save_tree(
    data: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> Path:
    if weights is None:
        xs = data[edges[:, 0]]
        ys = data[edges[:, 1]]
        weights = np.linalg.norm(xs - ys, axis=1)
    m = hashlib.sha512()
    m.update(edges[:,0].tobytes())
    m.update(edges[:,1].tobytes())
    m.update(weights.tobytes())
    digest = m.hexdigest()
    path = DATABASE_DIR / f"tree-{digest}.pq"
    tree = pl.DataFrame(dict(
        x=edges[:,0],
        y=edges[:,1],
        weight=weights
    ))
    tree.write_parquet(path)
    return path


def _run_ours(data, params, cluster: bool = False, cluster_k: int = 5):
    start = time.time()
    algo = panna.EMST(data, **params)
    elapsed_index_s = time.time() - start
    if cluster:
        tree_array, _core_array, _neighbors_array = algo.find_mst_dbscan(cluster_k)
        elapsed_discovery_s = time.time() - start - elapsed_index_s
        detail = dict(
            index_s=elapsed_index_s,
            discovery_s=elapsed_discovery_s,
            cluster_k=cluster_k,
        )
        detail |= algo.stats()
        edges = tree_array[:, :2].astype(np.int64)
        tree_weights = tree_array[:, 2]
        return edges, tree_weights, detail
    _, tree = algo.find_mst()
    elapsed_discovery_s = time.time() - start - elapsed_index_s
    detail = dict(index_s=elapsed_index_s, discovery_s=elapsed_discovery_s)
    detail |= algo.stats()
    return tree, None, detail


def _complete_tree(data, tree, core_distances):
    """Replace edges with infinite weight by arbitrary edges connecting
    different components, so that the result is a spanning tree with an
    actual (finite) weight. The replacement edges are weighted with the
    mutual reachability distance, like the ones already in the tree.
    Returns the fixed tree and the number of replaced edges."""
    infinite_mask = ~np.isfinite(tree[:, 2])
    if not infinite_mask.any():
        return tree, 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = data.shape[0]
    finite = tree[~infinite_mask]
    rows = finite[:, 0].astype(np.int64)
    cols = finite[:, 1].astype(np.int64)
    graph = coo_matrix((np.ones(len(finite)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)
    # pick an arbitrary representative for each component and connect
    # all of them to the representative of the first component
    _, representatives = np.unique(labels, return_index=True)
    anchor = representatives[0]
    others = representatives[1:]
    new_weights = np.maximum(
        np.linalg.norm(data[others] - data[anchor], axis=1),
        np.maximum(core_distances[others], core_distances[anchor]),
    )
    new_edges = np.column_stack(
        [
            np.full(len(others), anchor, dtype=np.float64),
            others.astype(np.float64),
            new_weights,
        ]
    )
    num_replaced = int(infinite_mask.sum())
    print(
        f"replacing {num_replaced} infinite-weight edges with "
        f"{len(others)} arbitrary edges connecting the components"
    )
    return np.vstack([finite, new_edges]), num_replaced


def _run_tutte(data, params):
    print("warmup")
    res = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(data[:10], **params)
    print("run tutte institute algorithm")
    res = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(data, **params)
    # `compute_minimum_spanning_tree` returns (mst_edges, neighbors,
    # core_distances), where the edge columns are [src, dst, mrd_weight]: the
    # weight is the mutual reachability distance, which coincides with the
    # Euclidean one only for min_samples=1. Report it as it is rather than
    # recomputing the distance between the endpoints, otherwise the recorded
    # weight is neither the weight of this tree nor the one of the EMST.
    tree, num_replaced = _complete_tree(data, res[0], res[2])
    edges = tree[:, :2].astype(np.int64)
    weights = tree[:, 2].astype(np.float64)
    return edges, weights, dict(replaced_infinite_edges=num_replaced)


def _run_pyhdbscan(data, params):
    import pyhdbscan

    print("run pyhdbscan (Wang-GFK) algorithm")
    min_pts = params.get("min_pts", 1)
    # pyhdbscan returns a single-linkage dendrogram as an (n-1, 4) array: columns
    # are [node, node, weight, size]. The weight column holds the (mutual
    # reachability) MST edge weights, which for min_pts=1 coincide with the
    # Euclidean EMST weights. Ids `>= n` denote clusters built by previous
    # merges, hence the first two columns are not point ids: go through
    # `_linkage_to_edges` to get a spanning tree over the points.
    res = np.asarray(pyhdbscan.HDBSCAN(data, min_pts))
    edges, weights = _linkage_to_edges(res, data.shape[0])
    return edges, weights, dict()


def _run_mlpack(data, params):
    import mlpack

    print("run mlpack algorithm")
    res = mlpack.emst(data)["output"]
    edges = res[:, :2].astype(np.int64)
    weights = res[:, 2].astype(np.float64)
    return edges, weights, dict()

# Default HNSW construction parameters, as used by the reference wrapper
# https://github.com/CamillaOkkels/singleLinkage-benchmark/blob/main/benchmark/algorithms/default_hnsw_params.py
HSSL_DEFAULT_PARAMS = dict(
    higher_max_degree=25,
    lowest_max_degree=50,
    max_layers=None,
    n_parallel_burnin=1_000,
    max_build_heap_size=100,
    max_build_frontier_size=None,
    level_norm_param_override=None,
    insert_heuristic=False,
    insert_heuristic_extend=False,
    post_prune_heuristic=False,
    insert_minibatch_size=None,
    n_rounds=1,
)



def _linkage_to_edges(linkage: np.ndarray, n: int):
    """Convert a scipy-style linkage matrix (rows `[node, node, weight, size]`,
    where ids `>= n` denote clusters created by previous merges) into a list of
    edges between point indices, plus the corresponding weights.

    Each merge is turned into an edge between arbitrary representatives of the
    two merged clusters: the result is a spanning tree of the points whose total
    weight equals the weight of the single linkage tree, but whose individual
    endpoints are *not* the endpoints of the corresponding MST edges (the
    dendrogram does not record them)."""
    num_merges = linkage.shape[0]
    representative = np.empty(n + num_merges, dtype=np.int64)
    representative[:n] = np.arange(n, dtype=np.int64)
    edges = np.empty((num_merges, 2), dtype=np.int64)
    for i, (a, b) in enumerate(linkage[:, :2].astype(np.int64)):
        edges[i, 0] = representative[a]
        edges[i, 1] = representative[b]
        representative[n + i] = representative[a]
    return edges, linkage[:, 2].astype(np.float64)


# self_join_neighbors = 100
# query_max_heap_size = 25
# higher_max_degree/lower_max_degree play the role of M
# query_max_heap_size is efSearch


def _run_hssl(data, params):
    import hnswhsslrust as hrr

    print("run hssl algorithm")
    data = np.ascontiguousarray(data, dtype=np.float32)

    start = time.time()
    min_pts = int(params.get("min_pts", 1))
    M = int(params.get("M", 100))
    efSearch = params.get("efS", None)
    efC = int(params.get("efC", 100))
    self_join_neighbors = params.get("self_join_neighbors", False)
    if self_join_neighbors is not None:
        res = hrr.hnsw_based_dendrogram_self_joined(
            data,
            min_pts,
            int(self_join_neighbors),
            int(efSearch),
            max_build_heap_size=efC,
            higher_max_degree=M,
            lowest_max_degree=M,
        )
    else:
        res = hrr.hnsw_based_dendrogram(
            data,
            min_pts=min_pts,
            max_build_heap_size=efC,
            higher_max_degree=M,
            lowest_max_degree=M,
        )
    elapsed_s = time.time() - start
    dendrogram = np.asarray(res[0])

    n = data.shape[0]
    if dendrogram.shape[0] != n - 1:
        print(
            f"warning: the dendrogram has {dendrogram.shape[0]} merges "
            f"instead of the expected {n - 1}"
        )
    edges, weights = _linkage_to_edges(dendrogram, n)
    detail = dict(
        hssl_s=elapsed_s,
    )
    return edges, weights, detail


def _run_ours_with_options(data, params, cluster, cluster_k):
    return _run_ours(data, params, cluster=cluster, cluster_k=cluster_k)


def worker(fn, fn_args, queue, emst_stats=False):
    start = time.time()
    res, tree_weights_override, detail = fn(*fn_args)
    end = time.time()
    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data = fn_args[0]
    if tree_weights_override is None:
        weight, tree_weights = tree_weight(data, res)
    else:
        tree_weights = np.asarray(tree_weights_override)
        weight = float(tree_weights.sum())
    print(f"algorithm completed, taking {end - start} seconds and {peak_memory_kb} kb")
    if emst_stats:
        print("computing statistics")
        diameter = panna.approximate_diameter(data)
        n = data.shape[0]
        npairs = n * (n - 1) // 2
        detail |= dict(diameter=float(diameter))
        # `compute_flexibility` and the threshold lookup below both assume
        # weights in ascending order. The diameter is only approximate (it is a
        # lower bound on the true one), hence single tree edges may be longer
        # than it: extend the histogram range so that no pair is left out.
        sorted_weights = np.sort(tree_weights)
        bounds, counts, mean_weight = compute_cumulative_distance_distribution(
            data, sorted_weights[0], max(diameter, sorted_weights[-1])
        )
        detail["mean_weight"] = float(mean_weight)
        detail["n"] = int(data.shape[0])
        detail["d"] = int(data.shape[1])
        for epsilon in [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]:
            flexibility = compute_flexibility(sorted_weights, epsilon, diameter)
            threshold = sorted_weights[-flexibility - 1]
            mass = compute_edge_mass(bounds, counts, threshold)
            contrast = estimate_contrast(mass, bounds, counts, diameter)
            detail |= {
                f"flexibility@{epsilon}": float(flexibility),
                f"mass@{epsilon}": float(mass),
                f"mass-frac@{epsilon}": float(mass / npairs),
                f"contrast@{epsilon}": float(contrast),
            }
    tree_path = save_tree(
        data,
        res,
        tree_weights if tree_weights_override is not None else None,
    )
    detail["tree_path"] = str(tree_path)

    _, detail_file_name = tempfile.mkstemp()
    with open(detail_file_name, "w") as fp:
        json.dump(detail, fp)

    # we have to pass back a file with the serialized detail, otherwise
    # the queue gets deadlocked because the size of the data is too large
    queue.put((weight, end - start, peak_memory_kb, detail_file_name))


def run_single(
    algorithm: str,
    dataset: str,
    parameters: dict,
    sample_frac: float | None,
    sample_seed: int = 1234,
    emst_stats: bool = False,
    cluster: bool = False,
    cluster_k: int = 5,
):
    dataset = Path(dataset).stem
    _, data = panna.datasets.load(
        dataset,
        pca_dimensions=4 if "pamap2" in dataset else None,
        normalize=any(d in dataset for d in ["angular", "cosine"]),
    )
    if sample_frac is not None:
        sample_size = int(sample_frac * data.shape[0])
        print(f"sampling {sample_size} elements")
        rng = np.random.default_rng(sample_seed)
        indices = rng.choice(data.shape[0], sample_size)
        data = data[indices]

    if cluster:
        parameters = {**parameters, "cluster_k": cluster_k}
    print(f"running {algorithm} on {dataset} with params {parameters} at sample fraction {sample_frac}")

    # algo_name = "k+scan" if algorithm == "k+" and cluster else algorithm
    algo_name = algorithm

    entry = Entry(
        algorithm=algo_name,
        parameters=parameters,
        dataset=dataset,
        dataset_sample_frac=sample_frac,
        dataset_sample_seed=sample_seed,
        dataset_sha=data_sha(data),
        **get_version(algo_name),
    )
    if already_run(entry.primary_key()):
        print(
            f"Configuration already run or running, skipping:\n\t{entry.primary_key()}"
        )
        return

    runners = {
        "k+": _run_ours_with_options,
        "tutte": _run_tutte,
        "pyhdbscan": _run_pyhdbscan,
        "mlpack": _run_mlpack,
        "hssl": _run_hssl,
    }
    if algorithm not in runners:
        raise ValueError(f"Unknown algorithm {algorithm}")

    runner = runners[algorithm]
    # spawn the algorithm as a subprocess, so that we can set a timeout and monitor
    # its memory usage. Use 'spawn' to avoid OpenMP-related fork issues.
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    if algorithm == "k+":
        runner_args = (data, parameters, cluster, cluster_k)
    else:
        runner_args = (data, parameters)
    proc = ctx.Process(target=worker, args=(runner, runner_args, queue, emst_stats))
    proc.start()
    proc.join(timeout=TIMEOUT_S)
    if proc.exitcode is None:
        # the process timed out, terminate it!
        print("Timeout!")
        proc.kill()
        # record a negative running time, to signal that the process
        # has been terminated forcibly after that many seconds
        entry.running_time_s = -TIMEOUT_S
    else:
        print("Process joined")
        assert proc.exitcode == 0
        emst_weight, elapsed_s, peak_memory_kb, detail_file_name = queue.get()
        with open(detail_file_name) as fp:
            detail = json.load(fp)
        Path(detail_file_name).unlink()
        # record the results
        entry.running_time_s = elapsed_s
        entry.memory_kb = peak_memory_kb
        entry.emst_weight = emst_weight
        entry.detail = detail


    # record the results by appending to the file and by recording
    # the detail in a parquet file by the side
    with FileLock(LOCKFILE):
        detail = entry.detail or {}
        if "profile" in detail:
            entry.profile_path = profile_sha_path(detail["profile"])
            profile = pl.DataFrame(detail["profile"])
            profile.write_parquet(entry.profile_path)
            del detail["profile"]
        entry.detail = detail

        with open(DATABASE_FILE, "a") as fp:
            line = json.dumps(sanitize_for_json(entry.as_dict()), allow_nan=False)
            fp.write(line + "\n")


ALGORITHMS = ["k+", "tutte", "pyhdbscan", "mlpack", "hssl"]


def run_experiments(
    datasets=None,
    algorithms=None,
    cluster: bool = False,
    cluster_k: int = 5,
    repetitions: int = 512,
):
    if datasets is None:
        import panna.datasets

        datasets = panna.datasets.available_datasets()
    if algorithms is None:
        algorithms = ["k+"]

    for dataset in datasets:
        # for sample_frac in [0.01, 0.1, 0.2, None]:
        for sample_frac in [None]:
            print(f"Running experiments on {dataset} at sample fraction {sample_frac}")
            if "k+" in algorithms:
                epsilons = [0.0, 0.1, 0.2, 0.5, 1.0]
                if cluster:
                    epsilons = [0.0, 1.0, 0.5]
                for epsilon in epsilons:
                    family = "lattice"
                    # if any(d in dataset for d in ["normalized", "angular", "cosine"]):
                    #     family = "crosspolytope"
                    params = {
                        "epsilon": epsilon,
                        "delta": 0.1,
                        "family": family,
                        "repetitions": repetitions,
                    }
                    if cluster and epsilon == 0.0:
                        params["refine_iterations"] = 1000
                    else:
                        params["refine_iterations"] = 0
                    run_single(
                        "k+",
                        dataset,
                        params,
                        sample_frac=sample_frac,
                        emst_stats=epsilon == 0.0,
                        cluster=cluster,
                        cluster_k=cluster_k,
                    )

            if "tutte" in algorithms:
                for exact in [True, False]:
                    tutte_params = {"min_samples": cluster_k if cluster else 1, "exact": exact}
                    run_single(
                        "tutte",
                        dataset,
                        tutte_params,
                        sample_frac=sample_frac
                    )
            if "pyhdbscan" in algorithms:
                pyhdbscan_params = {"min_pts": cluster_k if cluster else 1}
                run_single(
                    "pyhdbscan",
                    dataset,
                    pyhdbscan_params,
                    sample_frac=sample_frac
                )

            if "hssl" in algorithms:
                configs = []
                min_pts = cluster_k if cluster else 1
                for f in [2, 3, 4]:
                    for self_join_neighbors in [None, 100]:
                        M = max(20, f * min_pts)
                        efC = max(300, 1.5 * M)
                        efS_values = [5, 10]
                        if self_join_neighbors is not None:
                            for efS in efS_values:
                                hssl_params = {
                                    "min_pts": min_pts,
                                    "M": M,
                                    "efC": efC,
                                    "efS": efS,
                                    "self_join_neighbors": self_join_neighbors,
                                }
                                configs.append(hssl_params)
                        else:
                            hssl_params = {
                                "min_pts": min_pts,
                                "M": M,
                                "efC": efC,
                                "self_join_neighbors": self_join_neighbors,
                            }
                            configs.append(hssl_params)
                for hssl_params in configs:
                    run_single(
                        "hssl", dataset, hssl_params, sample_frac=sample_frac
                    )

            if "mlpack" in algorithms and not cluster:
                params = {}
                run_single(
                    "mlpack",
                    dataset,
                    params,
                    sample_frac=sample_frac
                )


def merge_results(other_file: Path):
    with open(DATABASE_FILE) as fp:
        current = set(fp.readlines())
    with open(other_file) as fp:
        new = [line for line in fp.readlines() if line not in current]
    print("Adding", len(new), "entries to the database")
    with open(DATABASE_FILE, "a") as fp:
        for line in new:
            fp.write(line)


def convert_results(path: Path):

    df = (
        pl.read_ndjson(path, infer_schema_length=None)
        .with_columns(
            profile_path=pl.col("detail")
            .struct.field("profile")
            .map_elements(profile_sha_path, return_dtype=pl.String)
        )
    )
    for profile, profile_path in df.select(
        pl.col("detail").struct.field("profile").alias("profile"), "profile_path"
    ).iter_rows():
        if profile_path is not None:
            profile = pl.DataFrame(profile)
            profile.write_parquet(profile_path)
        else:
            assert profile is None

    keep = [f.name for f in df.schema["detail"].fields if f.name != "profile"]
    converted = df.with_columns(
        pl.struct([pl.col("detail").struct.field(n) for n in keep]).alias("detail")
    )
    converted.write_ndjson(DATABASE_FILE)
    


def main():
    parser = argparse.ArgumentParser(description="EMST experiments script.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert data file")
    convert_parser.add_argument("file", type=Path, help="path to the ndjson file to convert")

    # run command
    run_parser = subparsers.add_parser("run", help="Run experiments.")
    run_parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset to run on. If not provided, all available datasets are used.",
    )
    run_parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS + ["all"],
        default="k+",
        help="Algorithm to run (default: k+, our own). Use 'all' to run every algorithm.",
    )
    run_parser.add_argument(
        "--cluster",
        action="store_true",
        help="Run the EMST clustering variant (uses find_mst_dbscan).",
    )
    run_parser.add_argument(
        "--cluster-k",
        type=int,
        default=5,
        help="Number of neighbors for the clustering variant (default: 5).",
    )
    run_parser.add_argument(
        "--repetitions",
        type=int,
        default=512,
        help="Number of repetitions for k+ (default: 512).",
    )

    # merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge another ndjson file into emst.json."
    )
    merge_parser.add_argument(
        "file",
        type=Path,
        help="Path to the ndjson file to merge.",
    )

    args = parser.parse_args()

    if args.command == "run":
        if args.dataset == "cluster" and not args.cluster:
            args.cluster = True
            args.dataset = None
        datasets_to_run = [args.dataset] if args.dataset else None
        if datasets_to_run:
            print(f"Running on specified datasets: {datasets_to_run}")
        else:
            print("Running on all available datasets.")
        algorithms = ALGORITHMS if args.algorithm == "all" else [args.algorithm]
        run_experiments(
            datasets_to_run,
            algorithms=algorithms,
            cluster=args.cluster,
            cluster_k=args.cluster_k,
            repetitions=args.repetitions,
        )
    elif args.command == "merge":
        merge_results(args.file)
    elif args.command == "convert":
        convert_results(args.file)


if __name__ == "__main__":
    main()
