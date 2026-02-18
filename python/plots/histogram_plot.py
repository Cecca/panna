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
    # Read the CSV file
    filepath_prefix = Path(__file__).resolve().parents[2]
    filepath = os.path.join(filepath_prefix, "histogram_nytimes.csv")
    df = pd.read_csv(filepath)
    # Drop the last row
    df = df[:-1]
    df = df[1:]
    # The mu is the sum of all counts until the :-2
    mu = df["count"][:-2].sum()
    print(f"Mu: {mu}")
    # For the rugplot we have to replicate the weights according to their counts
    expanded_weights = np.repeat(df["weight"], df["count"])
    expanded_df = pd.DataFrame({"weight": expanded_weights})

    sns.set_theme(palette="muted", style="white", font_scale=1.3)
    
    fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=1, layout="constrained")
    sns.histplot(data=expanded_df, x="weight", ax=axs)
    
    sns.rugplot(data=df, x="weight", ax=axs, height=-0.02,clip_on=False)
    axs.set_xlabel("Edge Weight")
    axs.set_ylabel("Count")
    sns.despine(left=True, bottom=True)
    plt.suptitle("Histogram of Edges found with no filter on fashion Dataset")
    #axs.set_xscale("log")
    
    plt.show()