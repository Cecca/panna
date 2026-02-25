import sys
from fast_hdbscan.hdbscan import compute_minimum_spanning_tree
import panna
import panna.datasets
import numpy as np
from icecream import ic
import time

def tree_weight(data, edges):
    xs = data[edges[:,0]]
    ys = data[edges[:,1]]
    ws = np.linalg.norm(xs - ys, axis=1)
    return ws.sum()

def tree_weight_mutual_reachability(data, edges, core_distances):
    xs = data[edges[:,0]]
    ys = data[edges[:,1]]
    ws = np.linalg.norm(xs - ys, axis=1)
    ic(core_distances, core_distances.min(), core_distances.max(), ws, ws.min(), ws.max())
    ws = np.maximum(ws, core_distances[edges[:,0]])
    ws = np.maximum(ws, core_distances[edges[:,1]])
    return ws.sum()
    

def check(dataset_name, sample_size=10000):
    rng = np.random.default_rng(1234)
    _, data = panna.datasets.load(dataset_name)
    rng.shuffle(data)
    data = data[:sample_size]

    start = time.time()
    algo = panna.EMST(data, epsilon=0.0, delta=0.01, repetitions=1024, family="lattice")
    _, emst = algo.find_mst()
    end = time.time()
    our_weight = tree_weight(data, emst.astype(np.int32))
    our_elapsed = end - start

    # _, exact_emst = algo.find_mst_exact()
    # exact_weight = tree_weight(data, exact_emst.astype(np.int32))

    start = time.time()
    baseline = compute_minimum_spanning_tree(data, min_samples=1)
    end = time.time()
    baseline_elapsed = end - start
    baseline_weight = tree_weight(data, baseline[0].astype(np.int32))
    success = abs(baseline_weight - our_weight) / baseline_weight < 1e-4
    success = our_weight <= baseline_weight + 1e-4
    return dict(
        dataset=dataset_name,
        success=success,
        our_elapsed=our_elapsed,
        baseline_elapsed=baseline_elapsed,
        our_weight=our_weight,
        # exact_weight=exact_weight,
        baseline_weight=baseline_weight,
    )


def check_mutual_reachability(dataset_name, knn=5, sample_size=10000):
    rng = np.random.default_rng(1234)
    _, data = panna.datasets.load(dataset_name)
    rng.shuffle(data)
    data = data[:sample_size]

    start = time.time()
    algo = panna.EMST(data, epsilon=0.0, delta=0.01, repetitions=1024, family="lattice")
    emst, core, _neighs = algo.find_mst_dbscan(knn)
    end = time.time()
    our_weight = tree_weight_mutual_reachability(data, emst[:,1:].astype(np.int32), core)
    our_elapsed = end - start

    # _, exact_emst = algo.find_mst_exact()
    # exact_weight = tree_weight(data, exact_emst.astype(np.int32))

    start = time.time()
    baseline = compute_minimum_spanning_tree(data, min_samples=knn)
    end = time.time()
    edges, _neighbors, baseline_core = baseline
    baseline_elapsed = end - start
    baseline_weight = tree_weight_mutual_reachability(data, baseline[0].astype(np.int32), baseline_core)
    success = abs(baseline_weight - our_weight) / baseline_weight < 1e-2
    # idxs = ic(np.argwhere(core - baseline_core > 0).flatten())
    # ic(np.linalg.norm(data[ic(_neighbors[idxs])][0] - data[idxs], axis=-1))
    return dict(
        dataset=dataset_name,
        success=success,
        our_elapsed=our_elapsed,
        baseline_elapsed=baseline_elapsed,
        our_weight=our_weight,
        baseline_weight=baseline_weight,
    )

results = [check(dataset) for dataset in ["glove-100-angular", "fashion-mnist-784-euclidean"]]
# results = [check(dataset) for dataset in ["ht", "fashion-mnist-784-euclidean", "glove-100-angular"]]
# results = []
results_mutual_reachability = [check_mutual_reachability(dataset) for dataset in ["fashion-mnist-784-euclidean"]]

overall_success = True
print("################# Summary ################")
for res in results:
    overall_success = overall_success and res["success"]
    mark = "✔️" if res["success"] else "❌"
    print(f"""{mark} {res["dataset"]}  elapsed={res["our_elapsed"]:.2f} (baseline {res["baseline_elapsed"]:.2f}) weight={res["our_weight"]} (baseline {res["baseline_weight"]})""")

print("################# Summary mutual reachability ################")
for res in results_mutual_reachability:
    overall_success = overall_success and res["success"]
    mark = "✔️" if res["success"] else "❌"
    print(f"""{mark} {res["dataset"]}  elapsed={res["our_elapsed"]:.2f} (baseline {res["baseline_elapsed"]:.2f}) weight={res["our_weight"]} (baseline {res["baseline_weight"]})""")

if not overall_success:
    sys.exit(1)
