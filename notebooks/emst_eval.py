import numpy as np
import polars as pl
import fast_hdbscan
# only `hdbscan` and `branches` are re-exported by the package, the rest of the
# submodules have to be asked for by name
from fast_hdbscan.cluster_trees import condense_tree, mst_to_linkage_tree
from collections.abc import Callable
import joblib
from pathlib import Path

MEM = joblib.Memory(".cache")


def load_tree(path: Path | str) -> np.ndarray:
    """Load a tree written by `save_tree` as a `(m, 3)` array of
    `[endpoint, endpoint, weight]` rows."""
    tree = pl.read_parquet(path)
    return np.column_stack(
        [
            tree["x"].to_numpy(),
            tree["y"].to_numpy(),
            tree["weight"].to_numpy(),
        ]
    ).astype(np.float64)


def load_base_tree(dataset, minPts) -> np.ndarray:
    from pathlib import Path
    import h5py

    path = None
    for d in Path("datasets").glob("*.hdf5"):
        if d.name.startswith(dataset):
            path = d
            break
    if path is None:
        raise ValueError("cannot find dataset")
    with h5py.File(path) as hfp:
        if f"/tree-{minPts}" not in hfp:
            raise ValueError(f"missing tree for {minPts}")
        return hfp[f"/tree-{minPts}/tree"][:]


def tree_clustering(tree: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Labels that HDBSCAN's condensation and excess-of-mass extraction assign
    to the hierarchy induced by `tree`, which is a `(n-1, 3)` array of
    `[endpoint, endpoint, weight]` rows. Noise points get the label -1."""
    labels, _, _, _, _ = fast_hdbscan.hdbscan.clusters_from_spanning_tree(
        np.asarray(tree, dtype=np.float64), min_cluster_size=min_cluster_size
    )
    return np.asarray(labels)


def compare_clusterings(
    reference_tree: np.ndarray,
    other_tree: np.ndarray,
    min_cluster_size: int = 10,
) -> dict:
    """Compare the HDBSCAN clusterings induced by two spanning trees of the
    same points, which must be indexed the same way in both trees.

    `reference_tree` is the one producing the reference clustering. Both ARI
    and AMI are symmetric, so swapping the arguments leaves them unchanged, but
    the diagnostics reported alongside them are per-tree.

    Noise points are compared as if they formed a cluster of their own, which
    is the convention that cannot be gamed by a tree that labels everything as
    noise. The agreement on which points are noise in the first place is
    reported separately as the Jaccard index of the two sets of noise points,
    together with how much of each clustering is noise: a high ARI/AMI paired
    with a low Jaccard index means the two trees agree on the cluster structure
    but disagree on its extent."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    reference_tree = np.asarray(reference_tree, dtype=np.float64)
    other_tree = np.asarray(other_tree, dtype=np.float64)
    if reference_tree.shape[0] != other_tree.shape[0]:
        raise ValueError(
            "the two trees have a different number of edges "
            f"({reference_tree.shape[0]} and {other_tree.shape[0]}): "
            "they cannot span the same set of points"
        )

    reference = tree_clustering(reference_tree, min_cluster_size)
    other = tree_clustering(other_tree, min_cluster_size)
    reference_noise = reference < 0
    other_noise = other < 0
    noise_union = int(np.count_nonzero(reference_noise | other_noise))

    return dict(
        min_cluster_size=int(min_cluster_size),
        ari=float(adjusted_rand_score(reference, other)),
        ami=float(adjusted_mutual_info_score(reference, other)),
        clusters_reference=int(np.unique(reference[~reference_noise]).shape[0]),
        clusters_other=int(np.unique(other[~other_noise]).shape[0]),
        noise_frac_reference=float(reference_noise.mean()),
        noise_frac_other=float(other_noise.mean()),
        noise_jaccard=(
            float(np.count_nonzero(reference_noise & other_noise) / noise_union)
            if noise_union > 0
            else 1.0
        ),
    )


def sweep_clusterings(
    reference_tree: np.ndarray,
    other_tree: np.ndarray,
    min_cluster_sizes=(5, 10, 25, 50, 100),
) -> list[dict]:
    """Run `compare_clusterings` at each of the given values of
    `min_cluster_size`. Agreement at a single value of `min_cluster_size` says
    little, since two trees can happen to agree at one granularity of the
    hierarchy and disagree at every other one."""
    return [
        compare_clusterings(reference_tree, other_tree, min_cluster_size=mcs)
        for mcs in min_cluster_sizes
    ]


def _complete_tree(data, tree):
    """Replace edges with infinite weight by arbitrary edges connecting
    different components, so that the result is a spanning tree with an
    actual (finite) weight. Returns the fixed tree and the number of
    replaced edges."""
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
    new_edges = np.column_stack(
        [
            np.full(len(others), anchor, dtype=np.float64),
            others.astype(np.float64),
            np.linalg.norm(data[others] - data[anchor], axis=1),
        ]
    )
    num_replaced = int(infinite_mask.sum())
    print(
        f"replacing {num_replaced} infinite-weight edges with "
        f"{len(others)} arbitrary edges connecting the components"
    )
    return np.vstack([finite, new_edges]), num_replaced


@MEM.cache
def _fast_hdbscan_tree(data: np.ndarray, min_samples: int) -> np.ndarray:
    tree = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(
        data, min_samples=min_samples
    )[0][:, :3]
    tree, _ = _complete_tree(data, tree)
    return np.asarray(tree, dtype=np.float64)


@MEM.cache
def noise_floor(
    data: np.ndarray,
    compare: Callable[[np.ndarray, np.ndarray], dict | list[dict]] | str,
    min_samples: int = 1,
    permutations: int = 3,
    seed: int = 1234,
) -> list[dict]:
    """What `compare` reports when `fast_hdbscan` is compared with *itself* on
    the same points in a permuted order.

    Ties among the distances make the minimum spanning tree non unique, and the
    extraction of the clusters is discontinuous in the merge heights, so two
    runs over the same points in a different order need not produce the same
    tree, let alone the same clustering. The scores collected here are the
    level at which a disagreement stops being distinguishable from tie
    breaking, and they are what makes any of the `compare_*` functions
    interpretable: an approximate tree scoring within this range is as close to
    the reference as the reference is to itself.

    `compare` is called as `compare(reference_tree, other_tree)` and may return
    either one record or a list of them, so any of `compare_clusterings`,
    `compare_cophenetic`, `compare_branches`, `compare_heights`,
    `compare_trees` and `sweep_clusterings` can be passed, on their own or
    wrapped to fix their parameters.

    Pass the `min_samples` that the trees under comparison were built with,
    otherwise the floor describes a different hierarchy than the one being
    scored."""
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    reference_tree = _fast_hdbscan_tree(data, min_samples)

    comparators = dict(
        cophenetic=compare_cophenetic
    )
    if isinstance(compare, str):
        compare = comparators[compare]

    records = []
    for permutation_index in range(permutations):
        permutation = rng.permutation(data.shape[0])
        tree = _fast_hdbscan_tree(data[permutation], min_samples)
        # the endpoints index the permuted rows: bring them back to the
        # indexing of the points that the reference tree uses
        tree[:, :2] = permutation[tree[:, :2].astype(np.int64)]
        reported = compare(reference_tree, tree)
        if isinstance(reported, dict):
            reported = [reported]
        records.extend(
            dict(permutation=permutation_index) | record for record in reported
        )
    return records


def linkage_tree(tree: np.ndarray) -> np.ndarray:
    """The linkage tree of a `(n-1, 3)` array of `[endpoint, endpoint, weight]`
    rows, as a SciPy-compatible `(n-1, 4)` matrix of
    `[node, node, height, size]` merges."""
    tree = np.asarray(tree, dtype=np.float64)
    return mst_to_linkage_tree(
        tree[np.argsort(tree[:, 2])]
    )


@MEM.cache
def cophenetic_heights(linkage: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Condensed vector of cophenetic heights for the points in `sample`: the
    merge height of the lowest common ancestor of each pair, ordered the way
    `scipy.spatial.distance.squareform` expects.

    For single linkage this height is the bottleneck distance between the two
    points, that is the largest edge on the path joining them in the spanning
    tree, so the vector describes the connectivity structure of the tree and
    not just the clustering that one cut through it produces.

    The lowest common ancestors are the ones of the whole tree, `sample` only
    selects which pairs get reported. The walk goes bottom up carrying the
    sampled points below each node, and writes every pair exactly once, when
    its two sides meet, so it costs O(n + s^2) instead of the O(n^2) that
    materialising the cophenetic matrix of every point would."""
    linkage = np.asarray(linkage, dtype=np.float64)
    num_points = linkage.shape[0] + 1
    sample = np.asarray(sample, dtype=np.int64)
    num_sampled = sample.shape[0]

    # position of each sampled point within `sample`, -1 for the others
    position = np.full(num_points, -1, dtype=np.int64)
    position[sample] = np.arange(num_sampled)

    heights = np.zeros(num_sampled * (num_sampled - 1) // 2, dtype=np.float64)
    members = {}

    def sampled_below(node: int) -> np.ndarray:
        if node < num_points:
            index = position[node]
            return np.empty(0, dtype=np.int64) if index < 0 else np.array([index])
        return members.pop(node)

    for index in range(linkage.shape[0]):
        left = sampled_below(int(linkage[index, 0]))
        right = sampled_below(int(linkage[index, 1]))
        if left.shape[0] and right.shape[0]:
            # every pair with one side left and the other right has its lowest
            # common ancestor here, so this merge is its cophenetic height
            low = np.minimum(left[:, None], right[None, :]).ravel()
            high = np.maximum(left[:, None], right[None, :]).ravel()
            heights[num_sampled * low - low * (low + 1) // 2 + high - low - 1] = (
                linkage[index, 2]
            )
        members[num_points + index] = np.concatenate([left, right])

    return heights


@MEM.cache
def compare_cophenetic(
    reference_tree: np.ndarray,
    other_tree: np.ndarray,
    sample_size: int = 2000,
    seed: int = 1234,
) -> dict:
    """Compare the ultrametrics that two spanning trees over the same points
    induce, on a random sample of the points.

    Two trees describe the same hierarchy exactly when their ultrametrics
    agree, so unlike `compare_clusterings` this scores every level at once and
    needs no cluster extraction in between, which makes it insensitive to the
    discontinuity of that extraction in the merge heights.

    The correlations are reported because they are the customary summary, but
    they are the least useful numbers here: most pairs of points merge near the
    root, at heights that any tree gets roughly right, and they dominate the
    coefficient. Prefer the errors, which are relative to the mean height and
    so comparable across datasets. `cophenetic_bias` is signed and should not
    come out negative: an approximate tree can only join points later than the
    exact one does, so points merging *earlier* means the reference is not
    exact."""
    from scipy.stats import rankdata

    reference_tree = np.asarray(reference_tree, dtype=np.float64)
    other_tree = np.asarray(other_tree, dtype=np.float64)
    if reference_tree.shape[0] != other_tree.shape[0]:
        raise ValueError(
            "the two trees have a different number of edges "
            f"({reference_tree.shape[0]} and {other_tree.shape[0]}): "
            "they cannot span the same set of points"
        )

    num_points = reference_tree.shape[0] + 1
    rng = np.random.default_rng(seed)
    sample = rng.choice(num_points, size=min(sample_size, num_points), replace=False)

    reference = cophenetic_heights(linkage_tree(reference_tree), sample)
    other = cophenetic_heights(linkage_tree(other_tree), sample)
    scale = reference.mean()

    return dict(
        sample_size=int(sample.shape[0]),
        cophenetic_pearson=float(np.corrcoef(reference, other)[0, 1]),
        cophenetic_spearman=float(
            np.corrcoef(rankdata(reference), rankdata(other))[0, 1]
        ),
        cophenetic_rmse=float(np.sqrt(np.mean((other - reference) ** 2)) / scale),
        cophenetic_mare=float(np.mean(np.abs(other - reference) / reference)),
        cophenetic_max_error=float(np.max(np.abs(other - reference)) / scale),
        cophenetic_bias=float(np.mean(other - reference) / scale),
    )


def branch_members(tree: np.ndarray, min_cluster_size: int) -> list[np.ndarray]:
    """The point sets of the branches of the condensed tree of `tree`.

    The raw linkage tree has one branch per merge, but single linkage chains,
    so almost all of them only add one point to the previous one and comparing
    them would mostly compare noise. Condensation keeps the branches that
    HDBSCAN considers real, which are the ones worth matching."""
    linkage = linkage_tree(tree)
    condensed = condense_tree(
        linkage, min_cluster_size=min_cluster_size
    )
    num_points = linkage.shape[0] + 1

    point_children = {}
    cluster_children = {}
    for parent, child in zip(condensed.parent, condensed.child):
        below = point_children if child < num_points else cluster_children
        below.setdefault(int(parent), []).append(int(child))

    members = {}
    # descending, so that a branch is assembled after all the ones below it
    branches = sorted(set(condensed.parent.tolist()), reverse=True)
    for branch in branches:
        parts = [np.array(point_children.get(branch, []), dtype=np.int64)]
        parts.extend(members[child] for child in cluster_children.get(branch, []))
        members[branch] = np.concatenate(parts)

    return [np.sort(members[branch]) for branch in branches]

@MEM.cache
def compare_branches(
    reference_tree: np.ndarray,
    other_tree: np.ndarray,
    min_cluster_size: int = 10,
) -> dict:
    """Compare the branches of the condensed trees of two spanning trees over
    the same points, by the best overlap that each branch of one finds among
    the branches of the other.

    This is the structural counterpart of `compare_cophenetic`: it ignores the
    merge heights and asks whether the same groups of points end up under the
    same branches. The Robinson-Foulds distance is the usual measure of that,
    but it counts a branch only when it is reproduced exactly, which nothing at
    this scale does; the Jaccard overlap degrades gracefully instead.

    The branch counts deserve as much attention as the scores. A tree that
    splits a reference branch in two keeps a high `branch_jaccard_weighted`,
    because the larger half still matches, and shows the damage only as a
    larger `branches_other`."""
    reference = branch_members(reference_tree, min_cluster_size)
    other = branch_members(other_tree, min_cluster_size)

    def best_overlaps(source, target) -> np.ndarray:
        target_sets = [set(branch.tolist()) for branch in target]
        overlaps = np.zeros(len(source))
        for index, branch in enumerate(source):
            branch = set(branch.tolist())
            overlaps[index] = max(
                (
                    len(branch & candidate) / len(branch | candidate)
                    for candidate in target_sets
                ),
                default=0.0,
            )
        return overlaps

    forward = best_overlaps(reference, other)
    backward = best_overlaps(other, reference)
    sizes = np.array([branch.shape[0] for branch in reference], dtype=np.float64)

    return dict(
        min_cluster_size=int(min_cluster_size),
        branches_reference=len(reference),
        branches_other=len(other),
        branch_jaccard=float((forward.mean() + backward.mean()) / 2),
        # the big branches are the ones the clustering is actually about
        branch_jaccard_weighted=float(np.average(forward, weights=sizes)),
        branch_recall_90=float(np.mean(forward >= 0.9)),
    )


def compare_heights(reference_tree: np.ndarray, other_tree: np.ndarray) -> dict:
    """Compare the multisets of merge heights of two spanning trees over the
    same points, which is the cheapest thing that can be said about them.

    The heights of a spanning tree are the deaths of the connected components
    in the filtration of the mutual reachability graph, and they all start at
    zero, so the Wasserstein distance between the two persistence diagrams is
    just the distance between the sorted vectors of edge weights and costs a
    sort. This says nothing about *which* points merge, so it cannot stand on
    its own, but it does separate a tree whose heights are wrong from one whose
    structure is wrong."""
    reference = np.sort(np.asarray(reference_tree, dtype=np.float64)[:, 2])
    other = np.sort(np.asarray(other_tree, dtype=np.float64)[:, 2])
    if reference.shape[0] != other.shape[0]:
        raise ValueError(
            "the two trees have a different number of edges "
            f"({reference.shape[0]} and {other.shape[0]}): "
            "they cannot span the same set of points"
        )

    scale = reference.mean()
    return dict(
        weight_ratio=float(other.sum() / reference.sum()),
        heights_wasserstein_1=float(np.mean(np.abs(other - reference)) / scale),
        heights_wasserstein_inf=float(np.max(np.abs(other - reference)) / scale),
    )


def compare_trees(
    reference_tree: np.ndarray,
    other_tree: np.ndarray,
    min_cluster_size: int = 10,
    sample_size: int = 2000,
    seed: int = 1234,
) -> dict:
    """Every comparison of this module at once, for one pair of trees.

    The three families answer different questions and are worth reading
    together: `compare_heights` asks whether the merge heights are right,
    `compare_cophenetic` whether the connectivity is, `compare_branches`
    whether the branch structure is, and `compare_clusterings` what all of that
    amounts to once the clusters are extracted. Right heights with wrong
    branches means the tree is fine and the extraction is merely
    discontinuous; right branches with wrong heights means the structure
    survives but the density estimates are off."""
    return (
        compare_heights(reference_tree, other_tree)
        | compare_cophenetic(
            reference_tree, other_tree, sample_size=sample_size, seed=seed
        )
        | compare_branches(
            reference_tree, other_tree, min_cluster_size=min_cluster_size
        )
        | compare_clusterings(
            reference_tree, other_tree, min_cluster_size=min_cluster_size
        )
    )


def tree_noise_floor(
    data: np.ndarray,
    min_cluster_size: int = 10,
    sample_size: int = 2000,
    min_samples: int = 1,
    permutations: int = 3,
    seed: int = 1234,
) -> list[dict]:
    """The `noise_floor` of `compare_trees`, so that every metric of this
    module comes with the level below which it says nothing.

    The floors are not alike: permuting the rows leaves the multiset of merge
    heights almost untouched, so `compare_heights` sits near zero and small
    differences there are real, while the branch counts move around freely and
    `compare_branches` needs a much wider band before a difference means
    anything."""
    return noise_floor(
        data,
        lambda reference, other: compare_trees(
            reference,
            other,
            min_cluster_size=min_cluster_size,
            sample_size=sample_size,
            seed=seed,
        ),
        min_samples=min_samples,
        permutations=permutations,
        seed=seed,
    )


if __name__ == "__main__":
    import sys
    import panna
    import json
    
    dataset = sys.argv[1]
    _, data = panna.datasets.load(
        dataset, normalize="angular" in dataset or "cosine" in dataset
    )
    ofile = "cophenetic-calibration.json"

    for core_k in [5, 15, 30, 60, 120, 240]:
        print(f"Calibrating for core_k={core_k}")
        calibration = noise_floor(
            data, "cophenetic", permutations=3, min_samples=core_k
        )
        with open(ofile, "a") as fp:
            for line in calibration:
                line |= dict(dataset=dataset, core_k=core_k)
                print(json.dumps(line), file=fp)
