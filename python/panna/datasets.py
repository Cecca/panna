"""
Datasets that can be used to try out different algorithms.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from urllib.parse import urlparse
import logging
import h5py
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse


DATASETS_DIR = Path(os.environ.get("PANNA_DATA_DIR", "datasets"))


def _download(url, destination: Path):
    import requests
    from tqdm import tqdm
    if not destination.is_file():
        logging.info(f"downloading {url} to {destination}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            with open(destination, "wb") as out_file, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    out_file.write(chunk)
                    progress.update(len(chunk))


def _load_hdf5(path: Path):
    with h5py.File(path) as hfp:
        return hfp["train"][:], hfp["test"][:], hfp["distances"][:]


def _load_pamap(path: Path):
    def load_zipfile():
        with zipfile.ZipFile(path, "r") as zip_ref:
            arr = []
            for i in range(1, 10):
                zfn = f"PAMAP2_Dataset/Protocol/subject10{i}.dat"
                zf = zip_ref.open(zfn)
                for line in zf:
                    line = line.decode()
                    l = list(map(float, line.strip().split()))
                    # remove timestamp and activity ID
                    arr.append(l[2:])
            X = np.nan_to_num(np.array(arr))  # many NaNs in data, replace them with 0.
            return X.astype(np.float32)

    h5path = path.parent / "pamap.hdf5"
    if not h5path.is_file():
        data = load_zipfile()
        with h5py.File(h5path, "w") as hfp:
            hfp["X"] = data

    with h5py.File(h5path) as hfp:
        data = hfp["X"][:]

    return data, None, None


def _load_census(path: Path):
    raw = np.load(path)
    data = raw["X"].astype(np.float32)
    return data, None, None

def _load_ht(path: Path):
    # Unzip
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(path.parent)
        # Unzip the inner file "HT_Sensor_dataset.zip"
        inner_zip_path = path.parent / "HT_Sensor_dataset.zip"
        # if there is an inner zip file, extract it
        if inner_zip_path.is_file():
            with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip_ref:
                inner_zip_ref.extractall(path.parent)
    # Load data
    data_path = path.parent / "HT_Sensor_dataset.dat"
    data = pd.read_csv(data_path, sep=r'\s+')
    del data["id"]
    data = data.to_numpy().astype(np.float32)
    data = np.nan_to_num(data)
    return data, None, None

def _load_chem(path: Path):
    # Unzip
    # Catch and handle not finding the file, try with the explicit file name
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(path.parent)
        # Load data
        data_path = path.parent / "gas+sensor+array+under+dynamic+gas+mixtures/ethylene_CO.txt"
        data = pd.read_csv(data_path, sep=r'\s+').to_numpy().astype(np.float32)
    except Exception as e:
        data_path = path.parent / "ethylene_CO.txt"
        data = pd.read_csv(data_path, sep=r'\s+').to_numpy().astype(np.float32)
    data = np.nan_to_num(data)
    return data, None, None
        

_DATASETS_INFO = {
    "landmark-nomic-768-normalized": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/landmark-nomic-768-normalized.hdf5?download=true",
        _load_hdf5,
        "euclidean",
    ),
    "imagenet-clip-512-normalized": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/imagenet-clip-512-normalized.hdf5?download=true",
        _load_hdf5,
        "euclidean",
    ),
    "agnews-mxbai-1024-euclidean": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/agnews-mxbai-1024-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "celeba-resnet-2048-cosine": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/celeba-resnet-2048-cosine.hdf5",
        _load_hdf5,
        "cosine",
    ),
    "simplewiki-openai-3072-normalized": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/simplewiki-openai-3072-normalized.hdf5?download=true",
        _load_hdf5,
        "euclidean",
    ),
    "deep-image-96-angular": (
        "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        _load_hdf5,
        "angular",
    ),
    "mnist-784-euclidean": (
        "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "fashion-mnist-784-euclidean": (
        "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "glove-100-angular": (
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        _load_hdf5,
        "angular",
    ),
    "gist-960-euclidean": (
        "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "nytimes-256-angular": (
        "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        _load_hdf5,
        "angular",
    ),
    "sift-128-euclidean": (
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "pamap2": (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
        _load_pamap,
        "euclidean",
    ),
    "census": (
        "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz",
        _load_census,
        "euclidean",
    ),
     "ht": (
         "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
            _load_ht,
            "euclidean",
    ),
     "chem": (
         "https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip",
         _load_chem,
         "euclidean",
    ),
}


def available_datasets():
    return list(_DATASETS_INFO.keys())


def _safe_l2_normalize_rows(data: np.ndarray, label: str) -> np.ndarray:
    """Normalize rows with L2 norm, keeping zero/invalid rows at zero."""
    out = np.array(data, dtype=np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)

    # Rows with non-finite or non-positive norm cannot be normalized safely.
    invalid_rows = (~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 0.0)
    invalid_count = int(np.count_nonzero(invalid_rows))
    if invalid_count > 0:
        logging.warning(
            "safe normalization for %s: %d rows had non-finite/non-positive norm and were set to zero",
            label,
            invalid_count,
        )

    np.divide(out, norms, out=out, where=(~invalid_rows)[:, np.newaxis])
    out[invalid_rows] = 0.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _array_health_stats(data: np.ndarray) -> tuple[int, int]:
    norms = np.linalg.norm(data, axis=1)
    zero_norm_rows = int(np.count_nonzero(norms <= 0.0))
    non_finite_values = int(np.count_nonzero(~np.isfinite(data)))
    return zero_norm_rows, non_finite_values


def load(
    name: str,
    pca_dimensions=None,
    center_mean=False,
    load_queries=False,
    normalize=False,
    standardize=False,
):
    if name not in available_datasets():
        raise KeyError(
            f"Dataset `{name}` not available. Pick one of {available_datasets()}"
        )
    if not DATASETS_DIR.is_dir():
        DATASETS_DIR.mkdir()

    url, loader, distance = _DATASETS_INFO[name]
    local_name = DATASETS_DIR / Path(urlparse(url).path).name
    _download(url, local_name)
    train, test, distances = loader(local_name)
    # Remove duplicate rows (if any) from train set
    train = np.unique(train, axis=0)
    # Remove completely NaN and infinite values from train and test sets, don't substitute with numbers
    train = train[~np.isnan(train).any(axis=1) & ~np.isinf(train).any(axis=1)]
    if test is not None:
        test = test[~np.isnan(test).any(axis=1) & ~np.isinf(test).any(axis=1)]
    
    if center_mean or standardize:
        scaler = StandardScaler(with_std=standardize)
        train = scaler.fit_transform(train)
        if test is not None:
            test = scaler.transform(test)

    if pca_dimensions is not None:
        pca = PCA(n_components=pca_dimensions)
        train = pca.fit_transform(train)
        if test is not None:
            test = pca.fit_transform(test)

    if normalize:
        train = _safe_l2_normalize_rows(train, "train")
        if test is not None:
            test = _safe_l2_normalize_rows(test, "test")

    # Remove duplicate rows that may have been (re)introduced by the transforms
    # above (e.g. PCA collapsing points, or normalization mapping collinear
    # vectors onto each other), regardless of the return path.
    train = np.unique(train, axis=0)

    if load_queries:
        return distance, train, test, distances
    else:
        return distance, train


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Load PANNA datasets and print dimensions/health diagnostics."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply safe L2 normalization before reporting stats.",
    )
    parser.add_argument(
        "--center-mean",
        action="store_true",
        help="Center features before reporting stats.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize features before reporting stats.",
    )
    args = parser.parse_args()

    print("name,train_rows,dimension,test_rows,train_zero_norm_rows,train_non_finite,test_zero_norm_rows,test_non_finite")
    for dataset_name in available_datasets():
        try:
            pca_dimensions = 4 if "pamap2" in dataset_name.lower() else None
            loaded = load(
                dataset_name,
                pca_dimensions=pca_dimensions,
                center_mean=args.center_mean,
                load_queries=True,
                normalize=args.normalize,
                standardize=args.standardize,
            )
            if len(loaded) != 4:
                raise RuntimeError(
                    f"unexpected loader output arity for {dataset_name}: {len(loaded)}"
                )
            _, train, test, _ = loaded

            train_zero, train_non_finite = _array_health_stats(train)
            if test is not None:
                test_zero, test_non_finite = _array_health_stats(test)
                test_rows = test.shape[0]
            else:
                test_zero, test_non_finite = 0, 0
                test_rows = 0

            print(
                f"{dataset_name},{train.shape[0]},{train.shape[1]},{test_rows},"
                f"{train_zero},{train_non_finite},{test_zero},{test_non_finite}"
            )
        except Exception as exc:
            print(f"{dataset_name},ERROR,{type(exc).__name__}:{exc}")
