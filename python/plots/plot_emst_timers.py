#!/usr/bin/env python3
"""
Parse EMST timer logs and plot a stacked bar chart of time spent in each
phase of the algorithm across different datasets.

Usage:
    python plot_emst_timers.py log1.txt log2.txt ...

Each log file should correspond to a run on a different dataset.  The dataset
name is inferred from the file name (e.g.  "sift.log" → "sift").

The script looks for lines containing  timer-name="emst_*"  and extracts the
elapsed_ms value.  All occurrences of each timer are summed (the Timer RAII
utility fires once per scope exit, so the same name appears many times).

The script auto-discovers which timers are present in the logs and groups
them into high-level categories for the stacked bar chart.  It handles both
old (coarse) and new (fine-grained) timer names transparently.

The output is a stacked bar chart saved as "emst_timing_breakdown.pdf".
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        "legend.markerscale": 1.5,
        "legend.handlelength": 1.5,
        "legend.handleheight": 1.0,
        "figure.titlesize": 12,
    }
)
sns.set_theme(font_scale=1.3, style="white", palette="muted")


# ── Category definitions ─────────────────────────────────────────────
# Each category has a label and a list of (timer_names, priority) groups.
# Within a category, if the fine-grained timers are present we use those;
# otherwise we fall back to the coarser one.
#
# "fine" timers take priority: if ANY fine timer for a category is found
# across all logs, we use only the fine timers for that category.
# Otherwise we fall back to the "coarse" timer.

CATEGORY_DEFS = [
    {
        "label": "Index scan + distances",
        "fine":  ["emst_bucket_scan", "emst_compute_distances"],
        "coarse": ["emst_scan_index"],
    },
    {
        "label": "Sort + merge",
        "fine":  ["emst_worker_sort", "emst_worker_collect",
                  "emst_collector_sort"],
        "coarse": [],
    },
    {
        "label": "Kruskal",
        "fine":  ["emst_worker_kruskal", "emst_collector_kruskal"],
        "coarse": ["emst_worker_mst", "emst_collector_mst"],
    },
    {
        "label": "Update filter",
        "fine":  ["emst_worker_filter_refresh", "emst_collect_edges"],
        "coarse": ["emst_collect_edges"],
    },
]

COLOURS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def parse_log_file(path: Path) -> dict[str, float]:
    """Return {timer_name: total_elapsed_ms} extracted from a single log file."""
    timers: dict[str, float] = defaultdict(float)
    pat = re.compile(r'timer-name="(emst_\w+)".*?elapsed_ms=(\d+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                timers[m.group(1)] += float(m.group(2))
    return dict(timers)


def dataset_label(path: Path) -> str:
    """Derive a human-readable dataset label from the log file name."""
    return path.stem.replace("_", " ").replace("-", " ")


def resolve_category_keys(cat: dict, all_keys: set[str]) -> list[str]:
    """Pick fine-grained timer keys if any are present, else coarse."""
    if any(k in all_keys for k in cat["fine"]):
        return cat["fine"]
    return cat["coarse"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot EMST timing breakdown across datasets."
    )
    parser.add_argument(
        "logs", nargs="+", type=Path,
        help="One or more log files, one per dataset.",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="emst_timing_breakdown.pdf",
        help="Output file name (default: emst_timing_breakdown.pdf).",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalise each bar to 100%% (proportions instead of absolute time).",
    )
    args = parser.parse_args()

    # ── collect data ──────────────────────────────────────────────────
    datasets: list[str] = []
    all_timers: list[dict[str, float]] = []

    for log_path in args.logs:
        if not log_path.exists():
            print(f"Warning: {log_path} not found, skipping.", file=sys.stderr)
            continue
        timers = parse_log_file(log_path)
        if not timers:
            print(f"Warning: no emst timers found in {log_path}, skipping.",
                  file=sys.stderr)
            continue
        datasets.append(dataset_label(log_path))
        all_timers.append(timers)

    if not datasets:
        print("No valid log files provided.", file=sys.stderr)
        sys.exit(1)

    # Diagnostic: show all discovered timers
    all_keys: set[str] = set()
    for t in all_timers:
        all_keys.update(t.keys())
    print("Discovered timer names:", sorted(all_keys), file=sys.stderr)

    n = len(datasets)

    # ── resolve categories ────────────────────────────────────────────
    resolved: list[tuple[str, list[str]]] = []
    for cat in CATEGORY_DEFS:
        keys = resolve_category_keys(cat, all_keys)
        if keys:
            resolved.append((cat["label"], keys))

    if not resolved:
        # Fallback: plot every timer except emst_total individually
        for k in sorted(all_keys):
            if k != "emst_total":
                resolved.append((k, [k]))

    print("Plot categories:", file=sys.stderr)
    for label, keys in resolved:
        print(f"  {label}: {keys}", file=sys.stderr)

    # ── build per-category arrays ─────────────────────────────────────
    # Raw sums (may exceed wall-clock because worker timers run in parallel)
    raw: dict[str, np.ndarray] = {}
    for label, raw_keys in resolved:
        raw[label] = np.array([
            sum(t.get(k, 0.0) for k in raw_keys) for t in all_timers
        ])

    totals = np.array([t.get("emst_total", 0.0) for t in all_timers])

    # Rescale: the sub-components keep their proportions but sum to
    # the wall-clock total, so the stacked bar never exceeds it.
    raw_sums = sum(raw[label] for label, _ in resolved)
    values: dict[str, np.ndarray] = {}
    for label, _ in resolved:
        values[label] = np.where(raw_sums > 0,
                                 raw[label] / raw_sums * totals, 0)

    if args.normalize:
        for label, _ in resolved:
            values[label] = np.where(totals > 0,
                                     values[label] / totals * 100, 0)
        ylabel = "Share of total time (%)"
    else:
        for label, _ in resolved:
            values[label] = values[label] / 1000.0
        totals = totals / 1000.0
        ylabel = "Time (s)"

    # ── plot ──────────────────────────────────────────────────────────
    colours = COLOURS[:len(resolved)]
    fig, ax = plt.subplots(figsize=(max(4, 1.2 * n + 2), 4.5))

    x = np.arange(n)
    bar_width = 0.55
    bottom = np.zeros(n)

    for (label, _), colour in zip(resolved, colours):
        ax.bar(x, values[label], bar_width, bottom=bottom,
               label=label, color=colour, edgecolor="white", linewidth=0.5)
        bottom += values[label]

    if not args.normalize:
        ax.scatter(x, totals, marker="_", s=120, color="black",
                   zorder=5, label="Total wall-clock")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("EMST timing breakdown")
    ax.legend(loc="upper left", fontsize="medium", framealpha=0.9)
    #ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    sns.despine()

    fig.savefig(args.output, dpi=400)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
