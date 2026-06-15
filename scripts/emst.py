#!/usr/bin/env python
"""
This script runs all the experiments regarding the EMST, including the baselines
"""

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
TIMEOUT_S = 24 * 3600 # 24 hours timeout


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
    return counts[idx]


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
        edges = tree_array[:, 1:3].astype(np.int64)
        tree_weights = tree_array[:, 0]
        return edges, tree_weights, detail
    _, tree = algo.find_mst()
    elapsed_discovery_s = time.time() - start - elapsed_index_s
    detail = dict(index_s=elapsed_index_s, discovery_s=elapsed_discovery_s)
    detail |= algo.stats()
    return tree, None, detail


def _run_tutte(data, params):
    print("run tutte institute algorithm")
    res = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(data, **params)
    return res[0].astype(np.int64), None, dict()


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
        bounds, counts, mean_weight = compute_cumulative_distance_distribution(
            data, tree_weights.min(), diameter
        )
        detail["mean_weight"] = float(mean_weight)
        detail["n"] = int(data.shape[0])
        detail["d"] = int(data.shape[1])
        for epsilon in [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]:
            flexibility = compute_flexibility(tree_weights, epsilon, diameter)
            threshold = tree_weights[-flexibility - 1]
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
        normalize="angular" in dataset,
    )
    if sample_frac is not None:
        sample_size = int(sample_frac * data.shape[0])
        print(f"sampling {sample_size} elements")
        rng = np.random.default_rng(sample_seed)
        indices = rng.choice(data.shape[0], sample_size)
        data = data[indices]

    if cluster:
        parameters = {**parameters, "cluster_k": cluster_k}

    algo_name = "k+scan" if algorithm == "k+" and cluster else algorithm

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
        "tutte": _run_tutte
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


def run_experiments(datasets=None, cluster: bool = False, cluster_k: int = 5):
    if datasets is None:
        import panna.datasets

        datasets = panna.datasets.available_datasets()

    for dataset in datasets:
        for sample_frac in [0.01, 0.1, 0.2, None]:
            print(f"Running experiments on {dataset} at sample fraction {sample_frac}")
            for epsilon in [0.0, 0.1, 0.2, 0.5, 1.0]:
                run_single(
                    "k+",
                    dataset,
                    {
                        "epsilon": epsilon,
                        "delta": 0.1,
                        "family": "lattice",
                        "repetitions": 512,
                    },
                    sample_frac=sample_frac,
                    emst_stats=epsilon == 0.0,
                    cluster=cluster,
                    cluster_k=cluster_k,
                )

            if sample_frac is not None:
                tutte_params = {"min_samples": 5 if cluster else 1}
                run_single(
                    "tutte",
                    dataset,
                    tutte_params,
                    sample_frac=sample_frac
                )


def merge_results(other_file: Path):
    import tempfile
    def append(input_fn: Path, output_fn: Path):
        with open(input_fn) as ifp:
            with open(output_fn, "a") as ofp:
                ofp.write(ifp.read())

    with FileLock(LOCKFILE):
        tmp = Path(tempfile.mkstemp()[1])
        append(DATABASE_FILE, tmp)
        append(other_file, tmp)

        df = pl.read_ndjson(tmp, infer_schema_length=None)
        # From Entry.primary_key()
        primary_keys = [
            "version",
            "algorithm",
            "parameters",
            "machine",
            "dataset",
            "dataset_sample_frac",
            "dataset_sample_seed",
            "dataset_sha",
        ]
        df_unique = df.unique(subset=primary_keys, keep="first")

        df_unique.write_ndjson(DATABASE_FILE)



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
        run_experiments(datasets_to_run, cluster=args.cluster, cluster_k=args.cluster_k)
    elif args.command == "merge":
        merge_results(args.file)
    elif args.command == "convert":
        convert_results(args.file)


if __name__ == "__main__":
    main()
