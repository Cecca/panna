import numpy as np
from pathlib import Path
from time import perf_counter
import sys
import os

sys.path.append(os.path.join(Path(__file__).resolve().parents[1]))

import panna
import argparse
from filelock import FileLock

if __name__ == "__main__":
    paths = [
        "fashion-mnist-784-euclidean.hdf5",
        "glove-100-angular.hdf5",
        "nytimes-256-angular.hdf5",
        "gist-960-euclidean.hdf5",
        "simplewiki-openai-3072-normalized.hdf5",
        "sift-128-euclidean.hdf5",
        "deep-image-96-angular.hdf5",
        "ethylene_CO.txt",
        "HT_Sensor_dataset.dat",
        "imagenet-align-640-normalized.hdf5",
        "landmark-nomic-768-normalized.hdf5",
        "9_census.npz",
        "PAMAP2_Dataset.zip",
    ]
    path_prefix = Path(__file__).resolve().parents[2]

    results_folder = os.path.join(path_prefix, "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Dataset filename to process (optional)")
    parser.add_argument("--knn", type=int, default=5, help="k for kNN in find_mst_dbscan (default: 5)")
    args = parser.parse_args()

    if args.path:
        paths = [args.path]

    for path in paths:
        stem = Path(path).stem
        _, data = panna.datasets.load(
            name=stem,
            pca_dimensions=4 if "pamap2" in stem.lower() else None,
            normalize=True if "chem" in stem.lower() else False,
        )
        data = np.array(data).astype(np.float32)#[:10000]

        algo = panna.EMST(data, epsilon=0.0, delta=0.05, repetitions=512, family="e2lsh")
        start_time = perf_counter()
        emst, core, _neighs = algo.find_mst_dbscan(args.knn)
        end_time = perf_counter()
        elapsed_time = end_time - start_time

        print(
            f"Finished dataset {path} | n={data.shape[0]} d={data.shape[1]} "
            f"knn={args.knn} | time={elapsed_time:.4f}s"
        )

        # lock_path = os.path.join(results_folder, "hdbscan_results.csv.lock")
        # out_path = os.path.join(results_folder, "hdbscan_results.csv")
        # with FileLock(lock_path):
        #     with open(out_path, "a+") as f_out:
        #         f_out.write(f"hdbscan, {data.shape[0]}, {path}, {args.knn}, {elapsed_time}\n")
        #         f_out.flush()
