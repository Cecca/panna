import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path
from matplotlib.gridspec import GridSpec

if __name__ == "__main__":
    mpl.use("WebAgg")
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
        }
    )

    filepath_prefix = Path(__file__).resolve().parents[2]
    filepath = os.path.join(filepath_prefix, "results/weight_results.csv")
    df = pd.read_csv(filepath)
    sns.set_theme(palette="muted", style="white", context="paper")

    df["Dataset"] = df["Dataset"].str.split("-").str[0]
    df["Dataset"] = df["Dataset"].str.replace("datasets/", "")
    df["group"] = df.groupby(["Dataset"])["Weight"].transform(lambda x: x.min())
    df["Relative Error"] = (df["Weight"] - df["group"]) / df["group"]
    df = df[df["Algorithm"] != "Exact"]
    deltas = sorted(df["delta"].unique())
    datasets = sorted(df["Dataset"].unique())

    # Rename algorithms for display
    algo_keys = ["K+", "K± e0.2", "K± e0.5", "K± e1", "K± e5", "K± e10", "K± e20"]
    algo_names = [
        r"K+",
        r"K+ $\varepsilon{=}0.2$",
        r"K+ $\varepsilon{=}0.5$",
        r"K+ $\varepsilon{=}1$",
        r"K+ $\varepsilon{=}5$",
        r"K+ $\varepsilon{=}10$",
        r"K+ $\varepsilon{=}20$",
    ]
    # Keep only algorithms present in data, preserving order
    present_keys = [k for k in algo_keys if k in df["Algorithm"].values]
    present_names = [algo_names[algo_keys.index(k)] for k in present_keys]
    algo_map = dict(zip(present_keys, present_names))
    df["Algorithm_display"] = df["Algorithm"].map(algo_map)

    n_algos = len(present_names)
    n_datasets = len(datasets)
    palette = sns.color_palette("muted", n_colors=n_datasets)
    markers = ["o", "s", "D", "^", "v", "P"]
    dodge_width = 0.22  # vertical offset between datasets within each algorithm row

    # Layout: one column per delta + narrow legend column
    n_deltas = len(deltas)
    fig = plt.figure(figsize=(12, 4), layout="constrained")
    gs = GridSpec(1, n_deltas + 1, width_ratios=[1] * n_deltas + [0.35], figure=fig)
    axs = [fig.add_subplot(gs[0, i]) for i in range(n_deltas)]
    ax_leg = fig.add_subplot(gs[0, n_deltas])
    ax_leg.axis("off")

    for col, delta in enumerate(deltas):
        ax = axs[col]
        subset = df[df["delta"] == delta]

        for k, ds in enumerate(datasets):
            ds_data = subset[subset["Dataset"] == ds]
            y_positions = []
            x_medians = []
            x_err_lo = []
            x_err_hi = []

            for j, algo_key in enumerate(present_keys):
                vals = ds_data[ds_data["Algorithm"] == algo_key]["Relative Error"]
                if len(vals) > 0:
                    med = vals.median()
                    x_medians.append(med)
                    x_err_lo.append(med - vals.quantile(0.25))
                    x_err_hi.append(vals.quantile(0.75) - med)
                    y_positions.append(j + (k - (n_datasets - 1) / 2) * dodge_width)

            ax.errorbar(
                x_medians, y_positions,
                xerr=[x_err_lo, x_err_hi],
                fmt=markers[k % len(markers)],
                color=palette[k],
                markersize=4,
                capsize=1.5,
                linewidth=1,
                label=ds if col == 0 else None,
            )

        ax.set_yticks(range(n_algos))
        if col == 0:
            ax.set_yticklabels(present_names)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_title(r"$\delta$ = " + f"{delta}")

        sns.despine(ax=ax)
        # Bound spines to data extent
        xmin = df["Relative Error"].min()
        xmax = df["Relative Error"].max()
        ax.spines["bottom"].set_bounds(xmin, xmax)
        ax.spines["left"].set_bounds(0, n_algos - 1)
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Light horizontal guides for readability
        for j in range(n_algos):
            ax.axhline(j, color="grey", linewidth=0.3, alpha=0.3, zorder=0)

    # Legend in the side panel
    handles, labels = axs[0].get_legend_handles_labels()
    ax_leg.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        title="Dataset",
    )

    fig.supxlabel("Relative Error (median, IQR)")

    plt.savefig("results/error_plot.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
