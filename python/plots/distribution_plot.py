import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path
import h5py


def distance(a, b):
    dot = np.dot(a, b)
    return np.sqrt(np.dot(a, a) + np.dot(b, b) - 2 * dot)


def norm(a):
    norm = np.sqrt(np.dot(a, a))
    return a / norm if norm != 0 else a


if __name__ == "__main__":
    mpl.use("WebAgg")
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "figure.titlesize": 12,
        }
    )
    sns.set_theme(palette="pastel", font_scale=1.3)
    sample_size = 1500
    filepath_prefix = Path(__file__).resolve().parents[2]
    datasets = [
        "fashion-mnist-784-euclidean.hdf5",
        "glove-100-angular.hdf5",
        "nytimes-256-angular.hdf5",
        "gist-960-euclidean.hdf5",
    ]

    fig, axs = plt.subplots(1, 4, layout="constrained")
    for dataset in datasets:
        print("Processing dataset:", dataset)
        filepath = os.path.join(filepath_prefix, dataset)
        with h5py.File(filepath, "r") as f:
            print("Reading data from:", filepath)
            data = f["train"][:]
            # Center the data and normalize it
            data = data - np.mean(data, axis=0)

            distances = []
            for _ in range(sample_size):
                idx1, idx2 = np.random.choice(data.shape[0], 2, replace=False)
                # normalize just the chosen vectors
                a = norm(data[idx1])
                b = norm(data[idx2])
                dist = distance(a, b)
                distances.append(dist)
            print("Finished distances")
            sns.kdeplot(
                distances,
                ax=axs[datasets.index(dataset)],
                fill=True,
                color="cornflowerblue",
                alpha=0.5,
            )
            axs[datasets.index(dataset)].set_title(dataset.split("-")[0])
            axs[datasets.index(dataset)].set_xlabel("Distance")
            axs[datasets.index(dataset)].set_ylabel("Density")
    # plt.tight_layout()
    plt.show()
