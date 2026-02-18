from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path

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

    filepath_prefix = Path(__file__).resolve().parents[2]
    filepath = os.path.join(filepath_prefix, "results/weight_results.csv")
    df_all = pd.read_csv(filepath)
    filepath = os.path.join(filepath_prefix, "results/toc_weight_results.csv")
    df_toc = pd.read_csv(filepath)
    sns.set_theme(palette="muted", style="white", context="paper", font_scale=1.3)

    # Clean up dataset names
    df_all["Dataset"] = df_all["Dataset"].str.split("-").str[0]
    df_all["Dataset"] = df_all["Dataset"].str.replace("datasets/", "")
    df_toc["Dataset"] = df_toc["Dataset"].str.split("-").str[0]
    df_toc["Dataset"] = df_toc["Dataset"].str.replace("datasets/", "")
    
    # Compute relative error
    df_all["group"] = df_all.groupby(["Dataset"])["Weight"].transform(lambda x: x.min())
    # Match the dataset groups
    for idx, row in df_toc.iterrows():
        dataset = row["Dataset"]
        min_weight = df_all[df_all["Dataset"] == dataset]["group"].iloc[0]
        # Compute relative error
        df_toc.at[idx, "Relative Error"] = (row["Weight"] - min_weight) / min_weight

    fig = plt.figure(figsize=(6, 3), layout="constrained")
    gs = GridSpec(1, 2, width_ratios=[1, 0.05], figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    palette = sns.color_palette("muted")
    algorithms = df_toc["Algorithm"].unique()
    color_map = dict(zip(algorithms, palette))

    bplot = sns.boxplot(
        data=df_toc,
        x="Dataset",
        y="Relative Error",
        hue="Algorithm",
        palette="muted",
        fill=False,
        boxprops=dict(alpha=0.7),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=2),
        showfliers=False,
        ax=ax,
        legend=True,
    )
    sns.despine(ax =ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.gca().spines["left"].set_bounds(df_toc["Relative Error"].min(), df_toc["Relative Error"].max())
    fig.supxlabel("Dataset")
    fig.supylabel("Relative Error")
    fig.suptitle("ToC Relative Error over Datasets")
    
    # Show the legend in a separate plot
    new_names = [
            r"ToC-K $\gamma$ 0.2 c 0.1",
            r"ToC-K $\gamma$ 0.5 c 0.1",
            r"ToC-K $\gamma$ 1.0 c 0.1",
            r"ToC-K $\gamma$ 0.05 c 0.05",
            r"ToC-K $\gamma$ 0.1 c 0.05",
            r"ToC-K $\gamma$ 0.2 c 0.05",
            r"ToC-K $\gamma$ 0.5 c 0.05",
            r"ToC-K $\gamma$ 1.0 c 0.05",
            r"ToC-K $\gamma$ 0.05 c 0.1",
            r"ToC-K $\gamma$ 0.1 c 0.1",
            r"ToC-K $\gamma$ 0.2 c 0.1",
            r"ToC-K $\gamma$ 0.5 c 0.1",
            r"ToC-K $\gamma$ 1.0 c 0.1",
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    ax_leg.legend(
        handles,
        new_names,
        loc="center",
    )


    plt.show()