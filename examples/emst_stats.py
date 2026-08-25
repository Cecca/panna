import pathlib
import panna
import panna.datasets
import numpy as np
import joblib
from icecream import ic
import csv
import logging
import matplotlib.pyplot as plt
import h5py


MEM = joblib.Memory(".cache")


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


def compute_emst(data):
    emst_algo = panna.EMST(data, epsilon=0.0, delta=0.1, repetitions=2048)
    emst = emst_algo.find_mst()
    return emst


def compute_stats_csv():
    logging.basicConfig(level=logging.INFO)

    outfile = pathlib.Path("emst_stats.csv")
    if not outfile.is_file():
        with open(outfile, "w") as fp:
            out = csv.writer(fp)
            out.writerow(
                ("dataset", "epsilon", "flexibility", "mass", "contrast", "num_pairs", "mass_fraction")
            )

    datasets = panna.datasets.available_datasets()
    for dataset in datasets:
        if dataset in ["chem", "deep-image-96-angular"]:
            continue
        ic(dataset)
        oname = pathlib.Path(f"{dataset}-stats.hdf5")
        if not oname.is_file():
            pca_dimensions = 4 if dataset == "pamap2" else None
            _, data = panna.datasets.load(
                dataset, pca_dimensions=pca_dimensions, normalize="angular" in dataset
            )
            n = data.shape[0]
            num_pairs = n * (n - 1) // 2
            weights, edges = compute_emst(data)
            ic(edges)
            diameter = panna.approximate_diameter(data)
            # the diameter is only approximate (it is a lower bound on the true
            # one), hence single tree edges may be longer than it: extend the
            # histogram range so that no pair is left out.
            bounds, counts, mean_weight = compute_cumulative_distance_distribution(
                data, weights.min(), max(diameter, weights.max())
            )

            with h5py.File(oname, "w") as hfp:
                hfp["/tree"] = edges
                hfp["/tree-weights"] = weights
                hfp["/weight-distribution/cumulative"] = counts
                hfp["/weight-distribution/bounds"] = bounds
                hfp.attrs["diameter"] = diameter
                hfp.attrs["mean_weight"] = mean_weight
                hfp.attrs["num_pairs"] = num_pairs
                hfp.attrs["num_points"] = n

        with h5py.File(oname) as hfp:
            edges = hfp["/tree"][:]
            weights = hfp["/tree-weights"][:]
            counts = hfp["/weight-distribution/cumulative"][:]
            bounds = hfp["/weight-distribution/bounds"][:]
            diameter = hfp.attrs["diameter"]
            num_pairs = hfp.attrs["num_pairs"]
            n = hfp.attrs["num_points"]
            mean_weight = hfp.attrs["mean_weight"]

        weights = np.sort(weights)
        ic(mean_weight, diameter)

        plt.figure()
        plt.title(dataset)
        plt.plot(bounds, counts)

        with open(outfile, "a") as fp:
            out = csv.writer(fp)
            for epsilon in [0, 0.01, 0.1]:
                flexibility = compute_flexibility(weights, epsilon, diameter)
                threshold = weights[-flexibility - 1]
                mass = compute_edge_mass(bounds, counts, threshold)
                contrast = estimate_contrast(mass, bounds, counts, diameter)
                plt.axhline(mass, linestyle="dotted")
                plt.axvline(threshold, linestyle="dotted")
                plt.annotate(f"ε={epsilon}", (0, mass))
                out.writerow(
                    (dataset, epsilon, flexibility, mass, contrast, num_pairs, mass / num_pairs)
                )
        plt.savefig(f"examples/cumdist-{dataset}.png", dpi=300)


compute_stats_csv()
