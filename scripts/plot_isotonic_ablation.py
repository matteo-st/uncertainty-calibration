#!/usr/bin/env python3
"""
Plot isotonic regression interpolation ablation results.

Generates a figure with 3 rows (ROCAUC, MCE, ECE) x 2 cols (SST-2, AG News)
for each score (SP, MD), comparing linear interpolation vs step function.

Also generates a combined 4-panel figure (2 scores x 2 datasets) for ROCAUC and MCE.

Usage:
    python scripts/plot_isotonic_ablation.py \
        --input results/isotonic_ablation/isotonic_ablation.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


COLORS = {
    "isotonic_linear": "#2ca02c",   # green
    "isotonic_step": "#d62728",     # red
}
LABELS = {
    "isotonic_linear": "Isotonic (linear interp.)",
    "isotonic_step": "Isotonic (step function)",
}
METHODS = ["isotonic_linear", "isotonic_step"]

DATASET_NAMES = {"sst2": "SST-2", "agnews": "AG News"}
SCORE_NAMES = {"sp": "SP", "md": "MD"}


def load_and_aggregate(json_path, score_filter=None):
    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    n_cal_values = sorted(data["config"]["n_cal_values"])

    agg = {}
    raw_rocaucs = {}

    for r in results:
        r_score = r.get("score", "md")
        if score_filter and r_score != score_filter:
            continue
        ds = r["dataset"]
        key = (r["method"], r["n_cal"], ds)
        if key not in agg:
            agg[key] = {"rocauc": [], "mce": [], "ece": [], "n_unique": [], "n_breakpoints": []}
        for m in ["rocauc", "mce", "ece", "n_unique", "n_breakpoints"]:
            if m in r and r[m] is not None and not (isinstance(r[m], float) and math.isnan(r[m])):
                agg[key][m].append(r[m])
        if r["method"] == "isotonic_linear":
            raw_rocaucs.setdefault(ds, []).append(r["raw_rocauc"])

    datasets = sorted(set(r["dataset"] for r in results))
    raw_rocauc_means = {ds: np.mean(vals) for ds, vals in raw_rocaucs.items()}

    return agg, n_cal_values, raw_rocauc_means, datasets


def plot_metric_on_ax(ax, agg, dataset, n_cal_values, metric,
                      show_raw_rocauc=False, raw_rocauc=None,
                      show_legend=False, show_xlabel=True):
    for method in METHODS:
        means, stds, valid_ncals = [], [], []
        for n_cal in n_cal_values:
            key = (method, n_cal, dataset)
            if key not in agg or not agg[key][metric]:
                continue
            valid_ncals.append(n_cal)
            means.append(np.mean(agg[key][metric]))
            stds.append(np.std(agg[key][metric]))

        valid_ncals = np.array(valid_ncals)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(valid_ncals, means, color=COLORS[method], marker="o",
                markersize=3, linewidth=1.2, label=LABELS[method], zorder=3)
        ax.fill_between(valid_ncals, means - stds, means + stds,
                        color=COLORS[method], alpha=0.15, zorder=2)

    if show_raw_rocauc and raw_rocauc is not None:
        ax.axhline(raw_rocauc, color="gray", linestyle="--", linewidth=1.0,
                   label="Raw (no cal.)", zorder=1)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(n_cal_values)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if show_xlabel:
        ax.set_xticklabels([str(v) for v in n_cal_values],
                           rotation=45, ha="right")
    else:
        ax.set_xticklabels([])

    if show_legend:
        loc = "lower right" if metric == "rocauc" else "upper right"
        ax.legend(loc=loc, fontsize=7, handlelength=1.5,
                  borderpad=0.3, labelspacing=0.3)


def plot_full_figure(json_path, output_path, score_filter):
    """3 rows (ROCAUC, MCE, ECE) x 2 cols (SST-2, AG News) for one score."""
    agg, n_cal_values, raw_rocauc_means, datasets = load_and_aggregate(
        json_path, score_filter=score_filter
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    dataset_order = ["sst2", "agnews"]
    metrics = ["rocauc", "mce", "ece"]
    metric_labels = {"rocauc": "ROCAUC", "mce": "MCE", "ece": "ECE"}

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))

    for col_idx, ds in enumerate(dataset_order):
        ds_name = DATASET_NAMES[ds]
        raw_roc = raw_rocauc_means.get(ds, 0.0)

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            is_last = (row_idx == len(metrics) - 1)
            is_first = (row_idx == 0 and col_idx == 0)

            plot_metric_on_ax(
                ax, agg, ds, n_cal_values, metric,
                show_raw_rocauc=(metric == "rocauc"),
                raw_rocauc=raw_roc,
                show_legend=is_first,
                show_xlabel=is_last,
            )

            if row_idx == 0:
                ax.set_title(ds_name, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_labels[metric])
            if is_last:
                ax.set_xlabel(r"$n_{\mathrm{cal}}$")

    fig.suptitle(f"Isotonic Regression: Linear vs Step — {SCORE_NAMES[score_filter]} score",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_combined_figure(json_path, output_path):
    """Combined figure: 3 metrics x 2 datasets, both scores overlaid."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    dataset_order = ["sst2", "agnews"]
    score_order = ["sp", "md"]
    metrics = ["rocauc", "mce", "ece"]
    metric_labels = {"rocauc": "ROCAUC", "mce": "MCE", "ece": "ECE"}

    # Colors per (method, score)
    colors = {
        ("isotonic_linear", "sp"): "#2ca02c",   # green
        ("isotonic_step", "sp"): "#98df8a",      # light green
        ("isotonic_linear", "md"): "#1f77b4",    # blue
        ("isotonic_step", "md"): "#aec7e8",      # light blue
    }
    linestyles = {
        "isotonic_linear": "-",
        "isotonic_step": "--",
    }

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))

    for score in score_order:
        agg, n_cal_values, raw_rocauc_means, datasets = load_and_aggregate(
            json_path, score_filter=score
        )

        for col_idx, ds in enumerate(dataset_order):
            raw_roc = raw_rocauc_means.get(ds, 0.0)

            for row_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]

                for method in METHODS:
                    means, stds, valid_ncals = [], [], []
                    for n_cal in n_cal_values:
                        key = (method, n_cal, ds)
                        if key not in agg or not agg[key][metric]:
                            continue
                        valid_ncals.append(n_cal)
                        means.append(np.mean(agg[key][metric]))
                        stds.append(np.std(agg[key][metric]))

                    valid_ncals = np.array(valid_ncals)
                    means = np.array(means)
                    stds = np.array(stds)

                    c = colors[(method, score)]
                    ls = linestyles[method]
                    mode_label = "linear" if method == "isotonic_linear" else "step"
                    label = f"{SCORE_NAMES[score]} {mode_label}"

                    ax.plot(valid_ncals, means, color=c, linestyle=ls,
                            marker="o", markersize=2.5, linewidth=1.2,
                            label=label, zorder=3)
                    ax.fill_between(valid_ncals, means - stds, means + stds,
                                    color=c, alpha=0.1, zorder=2)

                if metric == "rocauc" and score == "sp":
                    ax.axhline(raw_roc, color="gray", linestyle=":",
                               linewidth=0.8, zorder=1)

    for col_idx, ds in enumerate(dataset_order):
        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            n_cal_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
            ax.set_xticks(n_cal_values)
            ax.grid(True, alpha=0.3, linewidth=0.5)

            if row_idx == 2:
                ax.set_xticklabels([str(v) for v in n_cal_values],
                                   rotation=45, ha="right")
                ax.set_xlabel(r"$n_{\mathrm{cal}}$")
            else:
                ax.set_xticklabels([])

            if row_idx == 0:
                ax.set_title(DATASET_NAMES[ds], fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_labels[metric])

            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="lower right", fontsize=6.5,
                          ncol=2, handlelength=1.5)

    fig.suptitle("Isotonic Regression: Linear Interpolation vs Step Function",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_n_unique(json_path, output_path):
    """Plot number of unique output values vs n_cal."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    dataset_order = ["sst2", "agnews"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for score in ["sp", "md"]:
        agg, n_cal_values, _, _ = load_and_aggregate(json_path, score_filter=score)

        for col_idx, ds in enumerate(dataset_order):
            ax = axes[col_idx]
            for method in METHODS:
                means, valid_ncals = [], []
                for n_cal in n_cal_values:
                    key = (method, n_cal, ds)
                    if key not in agg or not agg[key]["n_unique"]:
                        continue
                    valid_ncals.append(n_cal)
                    means.append(np.mean(agg[key]["n_unique"]))

                c = {"isotonic_linear": "#2ca02c", "isotonic_step": "#d62728"}[method]
                ls = {True: "-", False: "--"}[score == "sp"]
                marker = "o" if score == "sp" else "s"
                mode = "linear" if method == "isotonic_linear" else "step"
                ax.plot(valid_ncals, means, color=c, linestyle=ls,
                        marker=marker, markersize=3, linewidth=1.2,
                        label=f"{SCORE_NAMES[score]} {mode}")

    for col_idx, ds in enumerate(dataset_order):
        ax = axes[col_idx]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(n_cal_values)
        ax.set_xticklabels([str(v) for v in n_cal_values], rotation=45, ha="right")
        ax.set_title(DATASET_NAMES[ds], fontweight="bold")
        ax.set_xlabel(r"$n_{\mathrm{cal}}$")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if col_idx == 0:
            ax.set_ylabel("Number of unique output values")
        if col_idx == 1:
            ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="results/isotonic_ablation/isotonic_ablation.json")
    parser.add_argument("--output_dir", type=str,
                        default="docs/paper/figures")
    args = parser.parse_args()

    fig_dir = args.output_dir

    # Per-score figures
    for score in ["sp", "md"]:
        plot_full_figure(args.input, f"{fig_dir}/isotonic_ablation_{score}.pdf", score)

    # Combined figure
    plot_combined_figure(args.input, f"{fig_dir}/isotonic_ablation_combined.pdf")

    # Unique values figure
    plot_n_unique(args.input, f"{fig_dir}/isotonic_ablation_nunique.pdf")


if __name__ == "__main__":
    main()
