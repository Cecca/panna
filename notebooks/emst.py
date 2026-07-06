# /// script
# dependencies = [
#     "altair==6.0.0",
#     "great-tables==0.21.0",
#     "marimo",
#     "numpy==2.4.4",
#     "polars==1.40.1",
#     "pyarrow==24.0.0",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.13"
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

    return cs, mo, pl, sns


@app.cell
def _():
    node_name = "lovelace" # pick only runs from the cluster
    return (node_name,)


@app.cell
def _():
    excluded_shas = [
        # the unnormalized glove-100 dataset
        "cfc5b3597505fbebf090fbeee98ec44efe4c1113c87b9d5919f89617167447c1d6b1d580fe5bd551f1a000bd278f8882d88ec560aa06b600f565db3277c8f9f3"
    ]
    return (excluded_shas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Baselines
    """)
    return


@app.cell
def _(cs, excluded_shas, node_name, pl, sizes):
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
            )
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
        .sort("running_time_s")

    )
    baselines
    return (baselines,)


@app.cell
def _():
    new = 55855880
    correct = 55845500
    relerr = abs(new-correct) / correct
    relerr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Our algorithms
    """)
    return


@app.cell
def _(mo):
    sel_algo_version = mo.ui.dropdown(
            ["0", "0.2.2", "1", "2", "3", "4", "5", "13"],
            label="algorithm version",
            value="13",
        )
    sel_algo_version
    return (sel_algo_version,)


@app.cell
def _(excluded_shas, mo, pl, sel_algo_version):
    full_results = (
        pl.read_ndjson("results/emst.json", infer_schema_length=None)
        .filter(pl.col("version") == sel_algo_version.value)
        .filter(pl.col("dataset_sha").is_in(excluded_shas).not_())
        .filter(
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
    )
    algo_name = mo.ui.dropdown.from_series(full_results["algorithm"], value="k+")
    algo_name
    return algo_name, full_results


@app.cell
def _(full_results, pl):
    emst_results = full_results.filter(pl.col("algorithm") == "k+")
    return (emst_results,)


@app.cell
def _(algo_name, full_results, pl):
    all_results = full_results.filter(pl.col("algorithm") == algo_name.value)
    all_results
    return (all_results,)


@app.cell
def _(all_results, pl):
    exact_full = (
        all_results.filter(pl.col("parameters").struct.field("epsilon") == 0.0)
        .filter(pl.col("dataset_sample_frac").is_null())
        .select("dataset", pl.col("running_time_s").round(2), "emst_weight")
    )
    exact_full
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
def _(dataset_list, dataset_stats, emst_results, node_name, pl):
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
        .filter(pl.col("algorithm") == "k+")
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
def _(approximate_full, baselines, cs):
    def approximate_normalized_time():
        our_cols = cs.matches(r"\d")
        base = baselines.pivot(index="dataset", on="algorithm", values="normalized_runtime")
        approximate_time_tbl = (
            approximate_full
            .pivot(index=["dataset"], on="epsilon", values=["normalized_runtime"], aggregate_function="mean")
            .join(base, on="dataset", how="left")
            .select("dataset", cs.exclude(our_cols, "dataset"), our_cols)
            .style
            # .fmt_number(columns=cs.numeric())
            .fmt_scientific(columns=cs.numeric())
            .tab_spanner(label="Our time (s)", columns=our_cols)
            # .tab_spanner(label="Baselines time (s)", columns=cs.exclude(our_cols, "dataset"))
            .cols_label_with(fn=lambda c: "epsilon=" + c, columns=our_cols)
        )

        with open("notebooks/emst-epsilon-table.tex", "w") as fp:
            print(approximate_time_tbl.as_latex().replace(r"epsilon=", r"$\epsilon=$"), file=fp)
        return approximate_time_tbl

    approximate_normalized_time()
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


@app.cell
def _(approximate_full, cs):
    approximate_full_tbl = (
        approximate_full
        .pivot(index=["dataset"], on="epsilon", values=["running_time_s", "relative_error", "distcomps_frac"], aggregate_function="mean")
        .style
        .fmt_number(columns=cs.starts_with("running_time"))
        .fmt_percent(columns=cs.starts_with("relative_error"))
        .fmt_percent(columns=cs.starts_with("distcomps"))
        .tab_spanner(label="Time (n)", columns=cs.starts_with("running_time"))
        .tab_spanner(label="Relative error", columns=cs.starts_with("relative_error"))
        .tab_spanner(label="Distance computations", columns=cs.starts_with("distcomps"))
        .cols_label_with(fn=lambda c: c.split("_")[-1], columns=cs.starts_with("running_time_s"))
        .cols_label_with(fn=lambda c: c.split("_")[-1], columns=cs.starts_with("relative_error"))
        .cols_label_with(fn=lambda c: c.split("_")[-1], columns=cs.starts_with("distcomps"))
    )

    with open("notebooks/emst-epsilon.tex", "w") as fp:
        print(approximate_full_tbl.as_latex(), file=fp)
    approximate_full_tbl
    return


@app.cell
def _(all_results, mo):
    sel_dataset = mo.ui.dropdown(all_results["dataset"].unique().to_list(), label="dataset", value="glove")
    sel_epsilon = mo.ui.dropdown(all_results["parameters"].struct.field("epsilon").unique().to_list(), label="epsilon", value=0.0)
    mo.vstack([
        sel_dataset,
        sel_epsilon
    ])
    return sel_dataset, sel_epsilon


@app.cell(disabled=True)
def _(all_results, pl, sel_dataset, sel_epsilon):
    profile_path = all_results.filter(
        pl.col("dataset") == sel_dataset.value,
        pl.col("parameters").struct.field("epsilon") == sel_epsilon.value,
        pl.col("dataset_sample_frac").is_null()
    ).select("profile_path").to_dicts()[0]["profile_path"]
    profile = pl.read_parquet(profile_path).with_columns(elapsed_s = pl.col("elapsed_ms") / 1000)
    (
        profile.select("elapsed_s", "emst_confirmed_weight", "emst_weight_lower_bound", "emst_total_weight")
            .unpivot(index="elapsed_s", variable_name="type", value_name="weight")
            .plot
            .line(x="elapsed_s", y="weight", color="type")
    )
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
