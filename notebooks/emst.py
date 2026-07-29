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
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.15"
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
        .filter(pl.col("version").is_in(["0.3.2", "13"]))
        .with_columns(
            pl.col("dataset").str.replace(
                "-[0-9]+-(euclidean|angular|normalized)", ""
            ),
            pl.when(pl.col("parameters").struct.field("exact"))
            .then(pl.col("algorithm") + "-exact")
            .otherwise("algorithm")
            .alias("display algorithm"),
        )
        .with_columns(
            pl.when(pl.col("algorithm") == "k+")
            .then(
                pl.lit("k+ (")
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
            .struct.field("cluster_k")
            .fill_null(pl.col("parameters").struct.field("min_samples"))
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
    return (baselines,)


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
    (
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
                    ["tutte-exact__", "tutte__"]
                    + ["k+__{}".format(e) for e in [0.0, 0.1, 0.2, 0.5, 1.0]]
                ),
            )
        )
        .tab_spanner(label="Ours", columns=cs.contains("k+"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("k+")
        )
        .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        .cols_label_with(
            fn=lambda c: "exact" if "exact" in c else "approx",
            columns=cs.contains("tutte"),
        )
        .fmt_number(columns=cs.numeric())
    )
    return


@app.cell
def _(GT, cs, pl, single_linkage_data):
    (
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
                    ["tutte-exact__", "tutte__"]
                    + ["k+__{}".format(e) for e in [0.0, 0.1, 0.2, 0.5, 1.0]]
                ),
            )
        )
        .tab_spanner(label="Ours", columns=cs.contains("k+"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("k+")
        )
        .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        .cols_label_with(
            fn=lambda c: "exact" if "exact" in c else "approx",
            columns=cs.contains("tutte"),
        )
        .fmt_percent(columns=cs.numeric(), decimals=4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mutual reachability clustering
    """)
    return


@app.cell
def _(experiments, pl):
    mutual_reachability_data = experiments.filter(pl.col("cluster_k") > 1)
    mutual_reachability_data.select(
        "dataset", "cluster_k", "display algorithm", "running_time_s", "relative_error"
    ).sort("dataset", "cluster_k", "display algorithm")
    return (mutual_reachability_data,)


@app.cell
def _(GT, cs, mutual_reachability_data, pl):
    (
        GT(
            mutual_reachability_data.with_columns(
                pl.when(pl.col("parameters").struct.field("exact"))
                .then(pl.col("algorithm") + "-exact")
                .otherwise("algorithm")
                .alias("algorithm")
            )
            .select(
                "dataset",
                "cluster_k",
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
            .select("dataset", "cluster_k", "pivot", "running_time_s")
            .pivot(
                on="pivot",
                index=["dataset", "cluster_k"],
                aggregate_function="min",
            )
            .select(
                "dataset", "cluster_k",
                pl.col(
                    ["tutte__"]
                    + ["k+__{}".format(e) for e in [0.5, 1.0]]
                ),
            )
            .sort("dataset", "cluster_k")
        )
        .tab_spanner(label="Ours", columns=cs.contains("k+"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("k+")
        )
        .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        .cols_label_with(
            fn=lambda c: "exact" if "exact" in c else "approx",
            columns=cs.contains("tutte"),
        )
        .fmt_number(columns=cs.numeric())
        .fmt_number(columns="cluster_k", decimals=0)
    )
    return


@app.cell
def _(GT, cs, mutual_reachability_data, pl):
    (
        GT(
            mutual_reachability_data.with_columns(
                pl.when(pl.col("parameters").struct.field("exact"))
                .then(pl.col("algorithm") + "-exact")
                .otherwise("algorithm")
                .alias("algorithm")
            )
            .select(
                "dataset",
                "cluster_k",
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
            .select("dataset", "cluster_k", "pivot", "relative_error")
            .pivot(
                on="pivot",
                index=["dataset", "cluster_k"],
                aggregate_function="min",
            )
            .select(
                "dataset", "cluster_k",
                pl.col(
                    ["tutte__"]
                    + ["k+__{}".format(e) for e in [0.5, 1.0]]
                ),
            )
            .sort("dataset", "cluster_k")
        )
        .tab_spanner(label="Ours", columns=cs.contains("k+"))
        .cols_label_with(
            fn=lambda c: c.split("__")[1], columns=cs.contains("k+")
        )
        .tab_spanner(label="Tutte", columns=cs.contains("tutte"))
        .cols_label_with(
            fn=lambda c: "exact" if "exact" in c else "approx",
            columns=cs.contains("tutte"),
        )
        .fmt_percent(columns=cs.numeric())
        .fmt_number(columns="cluster_k", decimals=0)
    )
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
def _(pl):
    mean_distances = (
        pl.read_csv("results/avg-weight.csv")
        .with_columns(
            pl.col("dataset").str.replace(
                "-[0-9]+-(euclidean|angular|normalized)", ""
            )
        )
        .select("dataset", "mean_distance")
    )
    return (mean_distances,)


@app.cell
def _(all_results, cs, mean_distances, pl, sizes):
    dataset_stats = (
        all_results.filter(pl.col("machine").struct.field("node_name") != "nixos")
        .filter(pl.col("parameters").struct.field("epsilon") == 0.0)
        .filter(pl.col("dataset_sample_frac").is_null())
        .unnest("detail")
        .filter(pl.col("flexibility@0.0").is_null().not_())
        .select(pl.exclude("n", "d"))
        .join(sizes, on="dataset", how="full")
        .with_columns(pl.when(pl.col("dataset").is_null()).then(pl.col("dataset_right")).otherwise(pl.col("dataset")).alias("dataset"))
        .join(mean_distances, on="dataset", how="left")
        .sort(pl.col("d"))
        .with_columns(tree_edges=pl.col("n") - 1)
        .with_columns(cs.starts_with("flexibility") / pl.col("tree_edges"))
        .select(pl.exclude("dataset_right"))
    )
    return (dataset_stats,)


@app.cell
def _(dataset_stats, pl):
    dataset_list = dataset_stats.sort(pl.col("d"))["dataset"].to_list()
    dataset_list
    return (dataset_list,)


@app.cell
def _(cs, dataset_stats, pl):
    dataset_stats_tbl = (
        dataset_stats
        .select("dataset", "n", "d",# pl.col("expected_cost"),
                "mass-frac@0.0", "mass-frac@0.5", "mass-frac@1.0",
                # "flexibility@0.5", "flexibility@1.0",
                "contrast@0.0", "contrast@0.5", "contrast@1.0")
        .sort(pl.col("d"))
        .style
        # .fmt_number(columns=["expected_cost"])
        .fmt_percent(columns=cs.starts_with("mass"))
        .fmt_percent(columns=cs.starts_with("flexibility"), decimals=3)
        .fmt_number(columns=["n", "d"], decimals=0)
        .fmt_number(columns=cs.starts_with("contrast"), decimals=2)
        .tab_spanner(label="Edge mass (percent)", columns=cs.starts_with("mass"))
        .cols_label_with(fn=lambda c: "ε=" + c.split("@")[-1], columns=cs.starts_with("mass"))
        # .tab_spanner(label="Edge flexibility", columns=cs.starts_with("flexibility"))
        # .cols_label_with(fn=lambda c: c.split("@")[-1], columns=cs.starts_with("flexibility"))
        .tab_spanner(label="Contrast", columns=cs.starts_with("contrast"))
        .cols_label_with(fn=lambda c: "ε=" + c.split("@")[-1], columns=cs.starts_with("contrast"))
    )

    with open("notebooks/dataset-stats.tex", "w") as _fp:
        print(dataset_stats_tbl.as_latex(), file=_fp)

    dataset_stats_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    """)
    return


@app.cell
def _(dataset_list, dataset_stats, emst_results, node_name, pl, sel_family):
    approximate_full = (
        emst_results
        .filter(pl.col("machine").struct.field("node_name") == node_name)
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("parameters").struct.field("repetitions") == 512)
        .join(dataset_stats, on="dataset", how="full")
        .with_columns(
            running_time_s = pl.when(pl.col("running_time_s") < 0).then(None).otherwise(pl.col("running_time_s"))
        )
        .with_columns(
            ground_weight = pl.col("emst_weight").min().over("dataset"),
            num_edges = pl.col("n") * (pl.col("n") - 1) / 2
        )
        .with_columns(
            weight_factor= pl.col("emst_weight") / pl.col("emst_weight").min().over("dataset"),
            relative_error = (pl.col("emst_weight") - pl.col("ground_weight")) / pl.col("ground_weight"),
            normalized_runtime = pl.col("running_time_s") / (pl.col("n") * pl.col("d"))
        )
        .with_columns(
            distcomps_frac = pl.col("detail").struct.field("distance_count") / pl.col("num_edges")
        )
        .filter(pl.col("parameters").struct.field("epsilon") >= 0.0)
        .select(pl.col("dataset"), 
                "algorithm", 
                pl.col("parameters")
                    .struct.field("repetitions").alias("repetitions"), 
                pl.col("parameters")
                    .struct.field("epsilon").alias("epsilon"),
                pl.col("running_time_s"),
                pl.col("relative_error"),
                pl.col("distcomps_frac"),
                pl.col("normalized_runtime")
        )
        .sort("dataset", "algorithm", "repetitions", "epsilon")
        .filter(pl.col("algorithm") == f"k+ ({sel_family.value})")
        .select("dataset", "epsilon", "running_time_s", "relative_error", "distcomps_frac", "normalized_runtime")
        .sort(
            pl.col("dataset").cast(pl.Enum(dataset_list))
        )
    )
    return (approximate_full,)


@app.cell
def _(approximate_full, dataset_list, sns):
    sns.barplot(
        data=approximate_full,
        y="dataset",
        x="running_time_s",
        hue="epsilon",
        order=dataset_list,
        errorbar=("ci", False)
    )
    return


@app.cell
def _(approximate_full, baselines, cs):
    def approximate_time():
        our_cols = cs.matches(r"\d")
        base = baselines.pivot(index="dataset", on="algorithm", values="running_time_s")
        approximate_time_tbl = (
            approximate_full
            .pivot(index=["dataset"], on="epsilon", values=["running_time_s"], aggregate_function="mean")
            .join(base, on="dataset", how="left")
            .select("dataset", cs.exclude(our_cols, "dataset"), our_cols)
            .style
            .fmt_number(columns=cs.numeric())
            .tab_spanner(label="Our time (s)", columns=our_cols)
            .tab_spanner(label="Baselines time (s)", columns=cs.exclude(our_cols, "dataset"))
            .cols_label_with(fn=lambda c: "$epsilon=" + c + "$", columns=our_cols)
        )

        with open("notebooks/emst-epsilon-table.tex", "w") as fp:
            print(to_latex(approximate_time_tbl), file=fp)
        return approximate_time_tbl

    approximate_time()
    return


@app.cell
def _(approximate_full, cs):
    def approximate_auxiliary():
        approximate_aux_tbl = (
            approximate_full
            .pivot(index=["dataset"], on="epsilon", values=["relative_error", "distcomps_frac"], aggregate_function="mean")
            .style
            .fmt_percent(columns=cs.starts_with("relative_error"))
            .fmt_percent(columns=cs.starts_with("distcomps"))
            .tab_spanner(label="Relative error", columns=cs.starts_with("relative_error"))
            .tab_spanner(label="Distance computations", columns=cs.starts_with("distcomps"))
            .cols_label_with(fn=lambda c: "epsilon=" + c.split("_")[-1], columns=cs.starts_with("relative_error"))
            .cols_label_with(fn=lambda c: "epsilon=" + c.split("_")[-1], columns=cs.starts_with("distcomps"))
        )
        with open("notebooks/emst-epsilon-aux.tex", "w") as fp:
            print(to_latex(approximate_aux_tbl), file=fp)
        return approximate_aux_tbl 

    approximate_auxiliary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On `census` there is a peculiar behavior in the plot above, which is probably due to the reindexing of the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering
    """)
    return


@app.cell
def _(excluded_shas, node_name, pl):
    experiments_clustering = (
        pl.read_ndjson("results/emst.json", infer_schema_length=None)
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("machine").struct.field("node_name") == node_name)
        .filter(pl.col("dataset_sha").is_in(excluded_shas).not_())
        .filter(pl.col("version").is_in(["0.3.2", "13"]))
        .filter(pl.col("parameters").struct.field("cluster_k").is_null().not_().or_(pl.col("parameters").struct.field("min_samples") > 1))
        .with_columns(
            pl.col("dataset").str.replace(
                "-[0-9]+-(euclidean|angular|normalized)", ""
            ),
            pl.when(pl.col("parameters").struct.field("exact"))
            .then(pl.col("algorithm") + "-exact")
            .otherwise("algorithm")
            .alias("display algorithm"),
            pl.col("parameters").struct.field("min_samples").fill_null(pl.col("parameters").struct.field("cluster_k")).alias("core_k")
        )
        .with_columns(
            pl.when(pl.col("algorithm") == "k+")
            .then(
                pl.lit("k+ (")
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
    )
    return (experiments_clustering,)


@app.cell
def _(GT, cs, experiments_clustering, pl):
    (
        GT(
            experiments_clustering.sort(
                "dataset", "core_k", "running_time_s"
            ).select(
                pl.format("{} ({})", "dataset", "core_k").alias("group"),
                "display algorithm",
                "running_time_s",
                "emst_weight",
            ),
            groupname_col="group",
        ).tab_options(row_group_as_column=True)
        .fmt_number(columns=cs.numeric())
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix: performance history
    """)
    return


@app.cell
def _(mo, pl):
    hist_data = (
        pl.read_ndjson("results/emst.json", infer_schema_length=None)
        # .filter(pl.col("dataset") == "sift")
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("parameters").struct.field("epsilon") == 0)
        .filter(pl.col("algorithm") == "k+")
    )
    hist_dataset = mo.ui.dropdown(hist_data["dataset"].unique().to_list())
    hist_dataset
    return hist_data, hist_dataset


@app.cell
def _(hist_data, hist_dataset, pl):
    (
        hist_data
        .filter(pl.col("dataset") == hist_dataset.value)
        .sort("timestamp")
        .select("timestamp", "version", "git_version", "running_time_s", "dataset_sha")
    )
    return


@app.cell
def _(hist_data, hist_dataset, pl):
    (
        hist_data
        .filter(pl.col("dataset") == hist_dataset.value)
        .sort("timestamp")
        .select("timestamp", "version", "git_version", "running_time_s", "dataset_sha")
    ).plot.line(x="timestamp", y="running_time_s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix: utilities
    """)
    return


@app.cell(disabled=True)
def _(all_results, pl):
    def download_trees(base="ceccarello@login.dei.unipd.it:/nfsd/lovelace/monaco/"):
        import subprocess as sp
        trees = (
            all_results
            .filter(pl.col("machine").struct.field("node_name") != "nixos")
            .filter(pl.col("parameters").struct.field("epsilon") == 0.0)
            .filter(pl.col("dataset_sample_frac").is_null())
            .unnest("detail")
            .filter(pl.col("flexibility@0.0").is_null().not_())
            .select("tree_path")["tree_path"].to_list()
        )
        for tree in trees:
            cmd = ["rsync", "--progress", base + tree, "results/"]
            sp.run(cmd)

    download_trees("algo:panna/")
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
