# /// script
# dependencies = [
#     "altair==6.0.0",
#     "fast-hdbscan==0.3.2",
#     "great-tables==0.21.0",
#     "h5py==3.16.0",
#     "marimo",
#     "numpy==2.4.6",
#     "polars==1.40.1",
#     "pyarrow==24.0.0",
#     "pynndescent==0.6.0",
#     "requests==2.34.2",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
#     "tqdm==4.70.0",
#     "umap-learn==0.5.12",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import great_tables
    from great_tables import GT
    import pyarrow
    import seaborn as sns
    import matplotlib.pyplot as plt
    import json

    return GT, cs, mo, np, pl, plt, sns


@app.cell
def _():
    node_name = "lovelace" # pick only runs from the cluster
    return (node_name,)


@app.cell
def _():
    excluded_shas = [
        # the unnormalized glove-100 dataset
        "cfc5b3597505fbebf090fbeee98ec44efe4c1113c87b9d5919f89617167447c1d6b1d580fe5bd551f1a000bd278f8882d88ec560aa06b600f565db3277c8f9f3",
        # HT
        "24158ef4c3bfbfb1d9f396591fdfca9211d84b15a4e182233096c23d665270b8436154aed0412688fd665fdf4d1d51da0dd1dab952aa311479a3c16ca2c414e5",
        # chem
        "276e488084569d335e232b50f5dc79abb62002b9f2882935b9a4ce20926ff3df9f9282863085ed378d7b559e48e046e2e62233ad028565e60a06bf5ec9db9710",
        # pamap2
        "c237bb7ffcd74ea885b10b8fc3b9f5309571f873d25359c11d7c598d9dde91022dfe9b69b324cdb6f88d35db8719aedc4d9ee752d1298cd59df3edd9855a4585",
        # census
        "9fb44a42de6d3412507051103a357630e2dda917013fb8cf03af298503bbffba794bf8c6e25dbb2623906464307925d3afdd704dff1c64ead63badb950f37a98",
        # pamap
        "3493a435ff8b9159507e99f2b373bc2c7f0b7d66d73b74222f1fb7f570e4a3bacec7511bf7d8aa8d7cef30902ef35b80ac7b29fe81a30720ee77bff6a5936bbc"
    ]
    return (excluded_shas,)


@app.cell
def _():
    datasets = ["sift", "mnist", "fashion-mnist", "glove", "nytimes"]
    return (datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Raw data loading
    """)
    return


@app.cell
def _(experiments, mo):
    dataset_sel = mo.ui.dropdown(sorted(experiments["dataset"].unique().to_list()), value="fashion-mnist")
    dataset_sel
    return (dataset_sel,)


@app.cell
def _(GT, dataset_sel, experiments, pl):
    GT(
        experiments
        .filter(pl.col("dataset") == dataset_sel.value)
        .filter(pl.col("cluster_k") == 1)    
        .select("dataset", "display algorithm", "running_time_s", "relative_error")
        .sort("dataset", "running_time_s")
    ).fmt_number(columns="running_time_s").fmt_number(columns="relative_error", decimals=5)
    return


@app.cell
def _(datasets, excluded_shas, ground_truth, node_name, pl):
    experiments = (
        pl.read_ndjson("results/emst.json", infer_schema_length=None)
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("machine").struct.field("node_name") == node_name)
        .filter(pl.col("dataset_sha").is_in(excluded_shas).not_())
        .filter(pl.col("version").is_in(["0.3.2", "13", "0.1.0", "4.8.0"]))
        .with_columns(
            pl.col("dataset").str.replace(
                "-[0-9]+-(euclidean|angular|normalized)", ""
            ),
            # the raw records still call the algorithm `k+`
            pl.col("algorithm").str.replace("k+", "panna", literal=True),
            pl.when(pl.col("parameters").struct.field("exact"))
            .then(pl.col("algorithm") + "-exact")
            .otherwise("algorithm")
            .alias("display algorithm"),
        )
        .with_columns(
            pl.when(pl.col("algorithm") == "panna")
            .then(
                pl.lit("panna (")
                + pl.col("parameters").struct.field("family").fill_null("?")
                + ", "
                + pl.col("parameters").struct.field("epsilon").fill_null("?")
                + pl.lit(")")
            )
            .when(pl.col("parameters").struct.field("exact"))
            .then(pl.col("algorithm") + "-exact")
            .otherwise(pl.col("algorithm"))
            .alias("display algorithm")
        )
        .with_columns(
            cluster_k=pl.col("parameters")
            .struct.field("min_pts")
            .fill_null(pl.col("parameters").struct.field("min_samples"))
            .fill_null(pl.col("parameters").struct.field("cluster_k"))
            .fill_null(pl.lit(1))
        )
        .join(ground_truth, on="dataset", how="left")
        .with_columns(
            relative_error=(
                pl.col("emst_weight") - pl.col("ground_weight")
            ).abs()
            / pl.col("ground_weight")
        )
        .select(pl.exclude("ground_weight"))
        .filter(pl.col("dataset").is_in(datasets))
    )
    return (experiments,)


@app.cell
def _(excluded_shas, pl):
    # Exact EMST weight per dataset, computed by the `k+` algorithm with
    # epsilon == 0. Used as the reference to compute the relative error of
    # the baselines.
    ground_truth = (
        pl.read_ndjson("results/emst.json", infer_schema_length=None)
        .filter(pl.col("algorithm").str.contains("k+"))
        .filter(pl.col("parameters").struct.field("epsilon") == 0.0)
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("dataset_sha").is_in(excluded_shas).not_())
        .filter(pl.col("parameters").struct.field("cluster_k").is_null())
        .filter( # keep most recent run
            pl.col("timestamp")
            == pl.col("timestamp")
            .max()
            .over(
                [
                    "algorithm",
                    "parameters",
                    "machine",
                    "dataset",
                    "dataset_sample_frac",
                    "dataset_sample_seed",
                ]
            )
        )
        .with_columns(
            pl.col("dataset").str.replace(
                "-[0-9]+-(euclidean|angular|normalized)", ""
            )
        )
        .group_by("dataset")
        .agg(ground_weight=pl.col("emst_weight").min())
    )
    return (ground_truth,)


@app.cell
def _(cs, excluded_shas, ground_truth, node_name, pl, sizes):
    baselines = (
        pl.read_ndjson("results/emst.json", infer_schema_length=None)
        .filter(pl.col("algorithm").str.contains("k+").not_())
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("machine").struct.field("node_name") == node_name)
        .filter(pl.col("dataset_sha").is_in(excluded_shas).not_())
        .filter(pl.col("version") == "0.3.2")
        .with_columns(
            pl.col("dataset").str.replace(
                "-[0-9]+-(euclidean|angular|normalized)", ""
            ),
            pl.when(pl.col("parameters").struct.field("exact")).then(pl.col("algorithm") + "-exact").otherwise("algorithm").alias("algorithm")
            # pl.when(pl.col("algorithm") == pl.lit("tutte")).then(pl.lit("tutte-") + pl.col("parameters").struct.field("exact")).otherwise("algorithm").alias("algoritm")
        )
        .join(sizes, on="dataset", how="inner")
        .with_columns(
            normalized_runtime=pl.col("running_time_s")
            / (pl.col("n") * pl.col("d"))
        )
        .select("dataset", "algorithm", "running_time_s", "memory_kb", "emst_weight")
        .with_columns(cs.numeric().round(2))
        .group_by("dataset", "algorithm")
        .agg(pl.col("*").mean())
        .join(ground_truth, on="dataset", how="left")
        .with_columns(
            relative_error=(
                pl.col("emst_weight") - pl.col("ground_weight")
            ).abs()
            / pl.col("ground_weight")
        )
        .select(pl.exclude("ground_weight"))
        .sort("running_time_s")

    )
    baselines
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single linkage clustering
    """)
    return


@app.cell
def _(experiments, pl):
    single_linkage_data = experiments.filter(pl.col("cluster_k") == 1)
    single_linkage_data.select(
        "dataset", "display algorithm", "running_time_s", "relative_error"
    )
    return (single_linkage_data,)


@app.cell
def _(GT, cs, pl, single_linkage_data):
    sl_time = (
        GT(
            single_linkage_data.with_columns(
                pl.when(pl.col("parameters").struct.field("exact"))
                .then(pl.col("algorithm") + "-exact")
                .otherwise("algorithm")
                .alias("algorithm")
            )
            .select(
                "dataset",
                "algorithm",
                pl.col("parameters").struct.field("epsilon"),
                "running_time_s",
            )
            .with_columns(
                (
                    pl.col("algorithm")
                    + "__"
                    + pl.col("epsilon").cast(pl.String).fill_null("")
                ).alias("pivot")
            )
            .sort("dataset", "algorithm", "epsilon")
            .select("dataset", "pivot", "running_time_s")
            .pivot(
                on="pivot",
                index="dataset",
                aggregate_function="min",
            )
            .select(
                "dataset",
                pl.col(
                    ["tutte-exact__", "tutte__", "mlpack__", "hssl__"]
                    + ["panna__{}".format(e) for e in [0.0, 0.1, 0.2, 0.5, 1.0]]
                ),
            )
        )
        .tab_spanner(label="HSSL", columns=cs.contains("hssl"))
        .cols_label_with(columns=cs.contains("hssl"), fn=lambda c: "approx")
        .tab_spanner(label="MLPACK", columns=cs.contains("mlpack"))
        .cols_label_with(columns=cs.contains("mlpack"), fn=lambda c: "exact")
        .tab_spanner(label="Ours", columns=cs.contains("panna"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("panna")
        )
        .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        .cols_label_with(
            fn=lambda c: "exact" if "exact" in c else "approx",
            columns=cs.contains("tutte"),
        )
        .fmt_number(columns=cs.numeric(), decimals=1)
    )
    with open("/tmp/single-linkage-time.tex", "w") as _fp:
        print(to_latex(sl_time), file=_fp)
    sl_time
    return


@app.cell
def _(GT, cs, pl, single_linkage_data):
    sl_error = (
        GT(
            single_linkage_data.with_columns(
                pl.when(pl.col("parameters").struct.field("exact"))
                .then(pl.col("algorithm") + "-exact")
                .otherwise("algorithm")
                .alias("algorithm")
            )
            .select(
                "dataset",
                "algorithm",
                pl.col("parameters").struct.field("epsilon"),
                "relative_error",
            )
            .with_columns(
                (
                    pl.col("algorithm")
                    + "__"
                    + pl.col("epsilon").cast(pl.String).fill_null("")
                ).alias("pivot")
            )
            .sort("dataset", "algorithm", "epsilon")
            .select("dataset", "pivot", "relative_error")
            .pivot(
                on="pivot",
                index="dataset",
                aggregate_function="min",
            )
            .select(
                "dataset",
                pl.col(
                    ["tutte-exact__", "tutte__", "mlpack__", "hssl__"]
                    + ["panna__{}".format(e) for e in [0.0, 0.1, 0.2, 0.5, 1.0]]
                ),
            )
        )    
        .tab_spanner(label="HSSL", columns=cs.contains("hssl"))
        .cols_label_with(columns=cs.contains("hssl"), fn=lambda c: "approx")
        .tab_spanner(label="MLPACK", columns=cs.contains("mlpack"))
        .cols_label_with(columns=cs.contains("mlpack"), fn=lambda c: "exact")
        .tab_spanner(label="Ours", columns=cs.contains("panna"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("panna")
        )
        .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        .cols_label_with(
            fn=lambda c: "exact" if "exact" in c else "approx",
            columns=cs.contains("tutte"),
        )
        .fmt_percent(columns=cs.numeric(), decimals=2)
    )
    with open("/tmp/single-linkage-error.tex", "w") as _fp:
        print(to_latex(sl_error), file=_fp)
    sl_error
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mutual reachability clustering
    """)
    return


@app.cell
def _(experiments, pl):
    mutual_reachability_data = experiments.filter(pl.col("cluster_k") > 1).select(pl.exclude("relative_error"))
    mutual_reachability_data.select(
        "dataset", "cluster_k", "display algorithm", "running_time_s"
    ).sort("dataset", "cluster_k", "display algorithm")
    return (mutual_reachability_data,)


@app.cell
def _(GT, cs, mutual_reachability_data, pl):
    mr_time = (
        GT(
            mutual_reachability_data.with_columns(
                pl.when(pl.col("parameters").struct.field("exact"))
                .then(pl.col("algorithm") + "-exact")
                .otherwise("algorithm")
                .alias("algorithm")
            )
            .with_columns(
                pl.col("parameters").struct.field("epsilon"),
                pl.col("parameters").struct.field("refine_iterations").fill_null(0)
            )
            .filter((pl.col("algorithm") != "panna").or_(((pl.col("refine_iterations") == 0).and_(pl.col("algorithm") == "panna"))))
            .with_columns(
                (
                    pl.col("algorithm")
                    + "__"
                    + pl.col("epsilon").cast(pl.String).fill_null("")
                ).alias("pivot")
            )
            .select("dataset", "cluster_k", "pivot", "running_time_s")
            .pivot(
                on="pivot",
                index=["dataset", "cluster_k"],
                aggregate_function="min",
            )
            .select(
                "dataset", "cluster_k",
                pl.col(
                    ["tutte__", "hssl__"]
                    + ["panna__{}".format(e) for e in [0.5, 1.0]]
                ),
            )
            .sort("dataset", "cluster_k"),
            # groupname_col="dataset", rowname_col="cluster_k"
        )
        .tab_spanner(label="Ours", columns=cs.contains("panna"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("panna")
        )
        .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        .cols_label_with(
            fn=lambda c: "exact" if "exact" in c else "approx",
            columns=cs.contains("tutte"),
        )
        .tab_spanner(label="HSSL", columns=cs.contains("hssl"))
        .cols_label_with(
            fn=lambda c: "",
            columns=cs.contains("hssl"),
        )
        .fmt_number(columns=cs.numeric())
        .fmt_number(columns="cluster_k", decimals=0)
        # .tab_options(row_group_as_column=True)
    )
    with open("/tmp/mr-time.tex", "w") as _fp:
        print(to_latex(mr_time), file=_fp)
    mr_time
    return


@app.cell
def _(GT, cs, mutual_reachability_data, pl):
    mr_error = (
        GT(
            mutual_reachability_data.with_columns(
                pl.when(pl.col("parameters").struct.field("exact"))
                .then(pl.col("algorithm") + "-exact")
                .otherwise("algorithm")
                .alias("algorithm")
            )
            .with_columns(
                pl.col("parameters").struct.field("epsilon"),
                pl.col("parameters").struct.field("refine_iterations").fill_null(0)
            )
            .filter(pl.col("refine_iterations") == 0)
            .with_columns(
                (
                    pl.col("algorithm")
                    + "__"
                    + pl.col("epsilon").cast(pl.String).fill_null("")
                ).alias("pivot")
            )
            .select("dataset", "cluster_k", "pivot", "emst_weight")
            .pivot(
                on="pivot",
                index=["dataset", "cluster_k"],
                aggregate_function="min",
            )
            .select(
                "dataset", "cluster_k",
                pl.col(
                    ["tutte__", "hssl__"]
                    + ["panna__{}".format(e) for e in [0.5, 1.0]]
                )
            )
            .with_columns((cs.contains("__") - pl.col("tutte__")) / pl.col("tutte__"))
            .select(pl.exclude("tutte__"))
            .sort("dataset", "cluster_k")
        )
        .tab_spanner(label="Ours", columns=cs.contains("panna"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("panna")
        )
        # .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        # .cols_label_with(
        #     fn=lambda c: "exact" if "exact" in c else "approx",
        #     columns=cs.contains("tutte"),
        # )
        .tab_spanner(label="HSSL", columns=cs.contains("hssl"))
        .cols_label_with(
            fn=lambda c: "",
            columns=cs.contains("hssl"),
        )
        .fmt_percent(columns=cs.numeric())
        .fmt_number(columns="cluster_k", decimals=0)
    )
    with open("/tmp/mr-error.tex", "w") as _fp:
        print(to_latex(mr_error), file=_fp)
    mr_error
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset statistics
    """)
    return


@app.cell
def _(pl):
    sizes = pl.DataFrame([
        {"dataset": "fashion-mnist", "n": 60000, "d": 784, "diameter": 5726.4541015625},
        {"dataset": "mnist", "n": 60000, "d": 784, "diameter": None},
        {"dataset": "gist", "n": 1000000, "d": 960, "diameter": 9.485732078552246},
        {"dataset": "glove", "n": 1183514, "d": 100, "diameter": 26.123464584350586},
        {"dataset": "imagenet-clip", "n": 1281167, "d": 512, "diameter": 1.4205868244171143},
        {"dataset": "landmark-nomic", "n": 760757, "d": 768, "diameter": 1.2240471839904785},
        {"dataset": "nytimes", "n": 290000, "d": 256, "diameter": 1.7088205814361572},
        {"dataset": "sift", "n": 1000000, "d": 128, "diameter": 719.6067504882812},
        {"dataset": "simplewiki-openai", "n": 260372, "d": 3072, "diameter": 1.5124},
        {"dataset": "ht", "n": 928991, "d": 11, "diameter": 378.7154541015625},
        {"dataset": "census", "n": 223223, "d": 500, "diameter": 11.313708305358887},
        {"dataset": "pamap2", "n": 2872533, "d": 4, "diameter": 663.171264648437},
        {"dataset": "chem", "n": 4208261, "d": 12, "diameter": 81767.921875},
        {"dataset": "deep-image", "n": 10000000, "d": 96, "diameter": None}
    ])
    return (sizes,)


@app.cell
def _(experiments, pl):
    (
        experiments
        .filter(pl.col("parameters").struct.field("epsilon") == 0)
        .select("dataset", "detail").unnest("detail")
        .select("dataset", "n", "d", "mass-frac@0.0", "contrast@0.0", "mass-frac@1.0", "contrast@1.0")
        .filter(pl.col("mass-frac@0.0") == pl.col("mass-frac@0.0").max().over("dataset"))
        .with_columns(
            cost = pl.col("n").pow(2/pl.col("contrast@0.0")) * (pl.col("mass-frac@0.0")*pl.col("n").pow(2)).pow(1-1/pl.col("contrast@0.0"))
        )
        .with_columns(
            pl.col("cost") / (pl.col("n").pow(2))
        )
        .sort("dataset")
    )
    return


@app.cell
def _(cs, experiments, pl):
    tbl_size = (
        experiments
        .filter(pl.col("parameters").struct.field("epsilon") == 0)
        .select("dataset", "detail").unnest("detail")
        .select("dataset", "n", "d", "mass-frac@0.0", "contrast@0.0", "mass-frac@1.0", "contrast@1.0")
        .filter(pl.col("mass-frac@0.0") == pl.col("mass-frac@0.0").max().over("dataset"))
        .sort("dataset")
        .style
        .fmt_number(columns=["n", "d"], decimals=0)
        .fmt_percent(columns=cs.contains("frac"))
        .fmt_number(columns=cs.contains("contrast"), decimals=4)
        .tab_spanner("$\\epsilon=0.0$", columns=cs.contains("0.0"))
        .tab_spanner("$\\epsilon=1.0$", columns=cs.contains("1.0"))
        .cols_label_with(
            fn=lambda c: "$\\mu_\\epsilon$",
            columns=cs.contains("mass"),
        )
        .cols_label_with(
            fn=lambda c: "contrast",
            columns=cs.contains("contrast"),
        )
    )
    with open("/tmp/sizes.tex", "w") as _fp:
        print(to_latex(tbl_size), file=_fp)
    tbl_size
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dendrogram agreement
    """)
    return


@app.cell
def _():
    from emst_eval import load_tree, sweep_clusterings, tree_clustering, compare_cophenetic, noise_floor

    return compare_cophenetic, load_tree, noise_floor


@app.cell
def _(noise_floor, pl):
    def cophenetic_calibration(trees):
        references = trees.filter(pl.col("algorithm") == "tutte")
        cases = references.select("dataset", "core_k").unique().to_dicts()
        res = []
        for case in cases[:3]:
            calibration = noise_floor(load_data(case["dataset"]), "cophenetic", permutations=5, min_samples=case["core_k"])
            res.extend(calibration)
        return pl.DataFrame(res)

    return


@app.cell
def _(excluded_shas, experiments, mo, pl):
    import os

    tree_catalogue = (
        experiments
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("dataset_sha").is_in(excluded_shas).not_())
        .filter(pl.col("detail").struct.field("tree_path").is_not_null())
        .select(
            pl.col("dataset").alias("full_dataset"),
            pl.col("dataset").str.replace(
                "-[0-9]+-(euclidean|angular|normalized)", ""
            ),
            "algorithm",
            pl.col("parameters").struct.field("epsilon"),
            pl.col("parameters")
            .struct.field("min_pts")
            .fill_null(pl.col("parameters").struct.field("min_samples"))
            .fill_null(pl.col("parameters").struct.field("cluster_k"))
            .fill_null(1)
            .alias("core_k"),
            pl.col("detail").struct.field("tree_path"),
            pl.col("parameters").struct.field("refine_iterations").fill_null(0)
        )
        .filter(pl.col("refine_iterations") == 0)
        .with_columns(
            local_path=pl.lit("results/")
            + pl.col("tree_path").str.split("/").list.last()
        )
        .unique()
        .with_columns(
            label=pl.col("algorithm")
            + pl.when(pl.col("epsilon").is_not_null())
            .then(
                pl.lit(" (ε=") + pl.col("epsilon").cast(pl.String) + pl.lit(")")
            )
            .otherwise(pl.lit(""))
        )
        # a tree that has not been downloaded yet cannot be compared
        .filter(
            pl.col("local_path").map_elements(
                os.path.exists, return_dtype=pl.Boolean
            )
        )
        .sort("dataset", "core_k", "algorithm", "epsilon")
    )
    tree_catalogue_input = mo.ui.table(tree_catalogue)
    tree_catalogue_input
    return (tree_catalogue,)


@app.cell
def _(compare_cophenetic, load_tree, mo, pl):
    @mo.cache
    def cophenetic_comparison(trees):
        references = trees.filter(pl.col("algorithm") == "tutte")
        datasets = references["dataset"].unique().to_list()
        res = []
        for dataset in datasets:
            cluster_ks = (
                references.filter(pl.col("dataset") == dataset)["core_k"]
                .unique()
                .to_list()
            )
            for ck in cluster_ks:
                filter_expr = (
                    pl.col("dataset") == dataset,
                    pl.col("core_k") == ck,
                )
                reference_tree = load_tree(
                    references.filter(filter_expr)["tree_path"].to_list()[0]
                )
                for experiment in trees.filter(filter_expr).to_dicts():
                    row = dict(dataset=dataset, core_k=ck, epsilon=experiment["epsilon"])
                    exp_tree = load_tree(experiment["tree_path"])
                    row["refine_iterations"] = experiment.get("refine_iterations", 0)
                    row["algorithm"] = experiment["algorithm"]
                    row |= compare_cophenetic(reference_tree, exp_tree)
                    res.append(row)

        return pl.DataFrame(res)

    return (cophenetic_comparison,)


@app.cell
def _(cophenetic_comparison, tree_catalogue):
    cophenetic_scores = cophenetic_comparison(tree_catalogue)
    cophenetic_scores
    return (cophenetic_scores,)


@app.cell
def _(GT, cophenetic_scores, cs, pl):
    mr_cophenetic = (
        GT(
            cophenetic_scores
            .filter(pl.col("algorithm") != "tutte")
            .with_columns(
                (
                    pl.col("algorithm")
                    + "__"
                    + pl.col("epsilon").cast(pl.String).fill_null("")
                ).alias("pivot")
            )
            .select("pivot", "dataset", "core_k", "cophenetic_mare")
            .pivot(
                on="pivot",
                index=["dataset", "core_k"],
                aggregate_function="min",
            )
            .select(
                "dataset",
                "core_k",
                pl.col(
                    ["hssl__"]
                    + ["panna__{}".format(e) for e in [0.5, 1.0]]
                ),
            )
            .sort("dataset", "core_k"),
            # groupname_col="dataset",
            # rowname_col="core_k",
        )
        .tab_spanner(label="Ours", columns=cs.contains("panna"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("panna")
        )
        .tab_spanner(label="HSSL", columns=cs.contains("hssl"))
        .cols_label_with(
            fn=lambda c: "",
            columns=cs.contains("hssl"),
        )
        .fmt_percent(columns=cs.numeric(), decimals=2)
        .fmt_number(columns="core_k", decimals=0)
        # .tab_options(row_group_as_column=True)
    )
    with open("/tmp/mr-cophenetic.tex", "w") as _fp:
        print(to_latex(mr_cophenetic), file=_fp)
    mr_cophenetic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix: utilities
    """)
    return


@app.cell
def _(pl):
    def download_trees(df, base="ceccarello@login.dei.unipd.it:/nfsd/lovelace/ceccarello/panna-tmp/"):
        """Downloads all the trees referenced in the given dataframe"""
        from pathlib import Path
        import subprocess as sp

        trees = (
            df
            .filter(pl.col("machine").struct.field("node_name") != "nixos")
            .select(pl.col("detail").struct.field("tree_path"))["tree_path"].to_list()
        )
        for tree in trees:
            if Path(tree).is_file():
                 continue
            cmd = ["rsync", "--progress", base + tree, "results/"]
            sp.check_call(cmd)

    return (download_trees,)


@app.cell
def _(download_trees, mutual_reachability_data):
    download_trees(mutual_reachability_data)
    return


@app.function
def to_latex(table):
    import re
    latex = table.as_latex()
    latex = latex.replace("epsilon", r"\epsilon")
    latex = re.sub(r"\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont", "", latex)
    latex = re.sub(r"\\(begin|end)\{table\}", "", latex)
    latex = re.sub(r"tabular\*", "tabular", latex)
    latex = re.sub(r"\{\\linewidth\}", "", latex)
    latex = latex.replace(r"[!t]", "")
    latex = latex.replace("None", "-")
    latex = latex.replace("\\$", "$")
    return latex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix: data investigations
    """)
    return


@app.function
def load_data(dataset):
    import sys
    sys.path.append("python/panna/")
    import datasets

    # the same loader the experiments use, so that the points, and hence
    # the hierarchy, are the ones the stored trees were built on
    dataset = [d for d in datasets.available_datasets() if d.startswith(dataset)][0]
    _, data = datasets.load(name=dataset)
    return data


@app.cell
def _(mo):
    @mo.cache
    def umap_embed(data):
        from umap import UMAP
        return UMAP().fit_transform(data)

    return (umap_embed,)


@app.cell
def _(np, plt, umap_embed):
    def view_clustering(data, tree, min_cluster_size):
        import fast_hdbscan
        prj = umap_embed(data)
        labels, probabilities, _, _, _ = fast_hdbscan.hdbscan.clusters_from_spanning_tree(
            np.asarray(tree, dtype=np.float64), min_cluster_size=min_cluster_size
        )
        tab10 = plt.get_cmap("tab10").colors
        _, ax = plt.subplots()
        # the noise goes down first, so that the clusters are drawn on top of it
        noise = labels == -1
        ax.scatter(
            prj[noise, 0],
            prj[noise, 1],
            s=1,
            c="#b0b0b0",
            linewidths=0,
            label="noise",
        )
        for i, label in enumerate(np.unique(labels[~noise])):
            sel = labels == label
            ax.scatter(
                prj[sel, 0],
                prj[sel, 1],
                size=3,
                color=tab10[i % len(tab10)],
                linewidths=0,
                label=str(label),
            )
        # the plotted markers are too small to be told apart in a legend, so
        # the handles get a size of their own
        legend = ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False
        )
        for handle in legend.legend_handles:
            handle.set_sizes([26])
        return ax

    return


@app.cell
def _(mo):
    mo.persistent_cache
    def compute_emst(data):
        import fast_hdbscan
        res = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(data, min_samples=1)[0]
        return res[:,:2], res[:,2]

    return (compute_emst,)


@app.cell
def _(compute_emst, mo, np, plt, sns):
    @mo.cache
    def edge_distribution(path):
        import h5py
        import pynndescent
        with h5py.File(path) as hfp:
            data = hfp["/train"][:]
            data = data[~np.all(data == 0, axis=1)]
            data = np.unique(data, axis=0)
            edges, weights = compute_emst(data)
        ax = sns.histplot(weights, bins=100)
        plt.show()
        return weights, ax

    return


if __name__ == "__main__":
    app.run()
