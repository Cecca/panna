from sklearn.decomposition import PCA
import h5py
import zipfile
import numpy as np
import pandas as pd
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

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Dataset filename to process (optional)")
    args = parser.parse_args()

    if args.path:
        paths = [args.path]

    for path in paths:
        # Use new dataset loader API. Pass pca_dimensions=4 for PAMAP2 dataset
        stem = Path(path).stem
        _, data = panna.datasets.load(name=stem, pca_dimensions=4 if 'pamap2' in stem.lower() else None, normalize=True if 'chem' in stem.lower() else False)
        data = np.array(data).astype(np.float32)
        family = "lattice"#"lattice" if data.shape[1] >= 20 else "e2lsh"

        if 'deep' in stem.lower(): # or 'pamap2' in stem.lower():
            elapsed_time = 0
        else: 
            emst = panna.EMST(data, delta=0.01, epsilon=0, family=family)
            start_time = perf_counter()
            emst.find_mst()
            end_time = perf_counter()
            elapsed_time = end_time - start_time
        #exit(0)

        emst = panna.EMST(data, delta=0.01, epsilon=0.2, family=family)
        start_time = perf_counter()
        emst.find_mst()
        end_time = perf_counter()
        elapsed_time2 = end_time - start_time
        print(f"Finished dataset {path}. Times: K+ {elapsed_time}, K± e0.2 {elapsed_time2}")

        # Write both results under a single file lock to avoid races from parallel jobs
        lock_path = os.path.join(results_folder, "scalability_results.csv.lock")
        out_path = os.path.join(results_folder, "scalability_results.csv")
        with FileLock(lock_path):
            with open(out_path, "a+") as f_out:
                f_out.write(f"K+, {data.shape[0]}, {path}, 0, {elapsed_time}\n")
                f_out.write(f"K± e0.2, {data.shape[0]}, {path}, 0, {elapsed_time2}\n")
                f_out.flush()