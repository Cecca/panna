import argparse
import numpy as np
from pathlib import Path
from time import perf_counter
import sys
import os

sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))
from panna.datasets import available_datasets, load

import pyhdbscan


def run_pyhdbscan(data, label):
    start = perf_counter()
    pyhdbscan.HDBSCAN(data, 1)
    elapsed = perf_counter() - start
    print(f"  pyhdbscan  {label}: {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pyhdbscan on datasets from panna.datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=available_datasets(),
        help=f"Dataset name(s). Available: {available_datasets()}",
    )
    parser.add_argument(
        "--random",
        nargs=2,
        type=int,
        metavar=("SIZE", "DIM"),
        action="append",
        default=[],
        help="Add a random dataset with given size and dimension (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/pyhdbscan_results.csv",
        help="CSV file to append results to",
    )
    args = parser.parse_args()

    results_folder = Path(__file__).resolve().parents[1] / "results"
    results_folder.mkdir(exist_ok=True)

    out_file = None
    if args.output:
        out_file = open(args.output, "a+")

    try:
        for name in args.datasets:
            print(f"Dataset: {name}")
            _, data = load(name, pca_dimensions= 4 if 'pamap2' in name.lower() else None, normalize=True if 'chem' in name.lower() else False)
            n, d = data.shape
            label = f"n={n} d={d}"
            t = run_pyhdbscan(data, label)
            if out_file:
                out_file.write(f"pyhdbscan,{name},{n},{d},{t}\n")

        for sz, dim in args.random:
            name = f"random-{sz}-{dim}"
            print(f"Dataset: {name}")
            data = np.random.rand(sz, dim).astype(np.float32)
            label = f"n={sz} d={dim}"
            t = run_pyhdbscan(data, label)
            if out_file:
                out_file.write(f"pyhdbscan,{name},{sz},{dim},{t}\n")

    finally:
        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()
