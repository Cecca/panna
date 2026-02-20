import fast_hdbscan
import numpy as np
from pathlib import Path
from time import perf_counter
import sys
import os
import argparse

sys.path.append(os.path.join(Path(__file__).resolve().parents[1]))

import panna


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tutte-Boruvka HDBSCAN on a dataset")
    parser.add_argument("--path", help="Dataset name to process (optional, runs all if omitted)")
    parser.add_argument(
        "--min-samples", type=int, default=1, help="min_samples for HDBSCAN (default: 1)"
    )
    args = parser.parse_args()

    paths = [
        "fashion-mnist-784-euclidean",
        "glove-100-angular",
        "nytimes-256-angular",
        "gist-960-euclidean",
        "simplewiki-openai-3072-normalized",
        "sift-128-euclidean",
        "deep-image-96-angular",
        "chem",
        "ht",
        "landmark-nomic-768-normalized",
        "census",
        "pamap2",
    ]

    if args.path:
        paths = [args.path]

    path_prefix = Path(__file__).resolve().parents[1]
    results_folder = os.path.join(path_prefix, "results")
    os.makedirs(results_folder, exist_ok=True)

    for name in paths:
        print(f"Loading dataset {name}...")
        _, data = panna.datasets.load(
            name=name,
            pca_dimensions=4 if "pamap2" in name.lower() else None,
            normalize=True if "chem" in name.lower() else False,
        )
        # Pick 25% of the data for testing
        data = np.array(data).astype(np.float32)[: len(data) // 100 * 10]

        print(f"Running Tutte-Boruvka HDBSCAN on {name} (n={data.shape[0]}, d={data.shape[1]})...")
        start_time = perf_counter()
        model = fast_hdbscan.HDBSCAN(min_samples=args.min_samples).fit(data)
        end_time = perf_counter()
        elapsed_time = end_time - start_time

        print(
            f"Finished dataset {name} | n={data.shape[0]} d={data.shape[1]} "
            f"min_samples={args.min_samples} | time={elapsed_time:.4f}s"
        )

        out_path = os.path.join(results_folder, "scalability_results_tutte.csv")
        with open(out_path, "a+") as f_out:
            f_out.write(
                f"Tutte-Boruvka, {data.shape[0]}, {name}, {args.min_samples}, {elapsed_time}\n"
            )
            f_out.flush()
