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
    sns.set_theme(palette="muted", style="white", font_scale=1.3)

    # Filepaths
    filepath_prefix = Path(__file__).resolve().parents[2]
    filepath_dim = os.path.join(filepath_prefix, "results/dimensionality_scalability.csv")
    filepath_len = os.path.join(filepath_prefix, "results/length_scalability.csv")

    # Color mapping 
    color_mapping = {
        "mlpack-Boruvka": "tab:orange",
        "K+": "tab:green",
        "K+ ɛ 0.1": "tab:red",
        "K+ 1 thread": "tab:purple",
        "Tutte-Boruvka": "tab:brown",
        "Wang-GFK": "tab:pink",
        "ToC-Kr": "tab:blue",
        "K+ ɛ 5.0": "tab:cyan",
    }

    # DIMENSIONALITY SCALABILITY PLOT
    df = pd.read_csv(filepath_dim)

    # Create a figure with an extra column for the legend
    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(1, 2, width_ratios=[6, 1], figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    sns.lineplot(
        data=df,
        x="D",
        y="Time (s)",
        hue="Algorithm",
        marker="o",
        ax=ax,
        palette=color_mapping,
        legend=True,
    )

    plt.suptitle("Dimensionality Scalability")
    plt.xlabel("Number of Dimensions (D)")
    plt.ylabel("Time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    sns.despine()

    # Tufte-style axis bounds
    ax.spines["left"].set_bounds(
        df[df["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].min(),
        df[df["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].max(),
    )
    ax.spines["bottom"].set_bounds(df["D"].min(), df["D"].max())

    # Create legend in the lateral subplot
    handles, labels = ax.get_legend_handles_labels()
    # Deactivate the legend in the main plot
    ax.legend_.remove()
    ax_leg.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        fontsize=9,
        title="Algorithm",
    )

    plt.tight_layout()
    #plt.show()
    fig.savefig("results/dimensionality_scalability_plot.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

    # LENGTH SCALABILITY PLOT
    df_length = pd.read_csv(filepath_len)
    dimensions = sorted(df_length["D"].unique())
    timeout = 28800  # 8 hours

    nrows = len(dimensions) // 2 + len(dimensions) % 2

    # Create grid with lateral legend column
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(nrows, 3, width_ratios=[1, 1, 0.3], figure=fig)
    axs = []
    for i, dim in enumerate(dimensions):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        axs.append(ax)
        subset = df_length[df_length["D"] == dim]

        sns.lineplot(
            data=subset,
            x="n",
            y="Time (s)",
            hue="Algorithm",
            marker="o",
            ax=ax,
            errorbar=None,
            palette=color_mapping,
            legend=False if i > 0 else True,
        )

        # Timeout dotted lines, add a cross at the timeout point
        for alg in subset["Algorithm"].unique():
            alg_data = subset[subset["Algorithm"] == alg]
            max_n = alg_data["n"].max()
            if max_n < 10**6 and alg_data["Time (s)"].max() < timeout:
                ax.plot(
                    [max_n, max_n * 10],
                    [alg_data["Time (s)"].max(), timeout],
                    linestyle="dotted",
                    color=color_mapping.get(alg, "gray"),
                )
                ax.plot(
                    max_n * 10,
                    timeout,
                    marker="P",
                    color=color_mapping.get(alg, "gray"),
                )

        ax.set_title(f"D = {dim}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        sns.despine(ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.spines["left"].set_bounds(
            subset[subset["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].min(),
            subset[subset["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].max(),
        )
        ax.spines["bottom"].set_bounds(subset["n"].min(), subset["n"].max())

    # Common labels
    fig.supylabel("Time (s)")
    fig.supxlabel("Number of Points (n)")
    fig.suptitle("Length Scalability by Dimension")

    # Legend subplot
    ax_leg = fig.add_subplot(gs[:, 2])
    ax_leg.axis("off")
    handles, labels = axs[0].get_legend_handles_labels()
    # Deactivate the legend in the main plot
    axs[0].legend_.remove()
    ax_leg.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        fontsize=9,
        title="Algorithm",
    )

    plt.tight_layout()
    #plt.show()
    fig.savefig("results/length_scalability_plot.pdf", dpi=600, bbox_inches="tight", format="pdf")
    plt.close(fig)
