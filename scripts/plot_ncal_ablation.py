#!/usr/bin/env python3
"""
Plot n_cal ablation results: MCE, ECE, and ROCAUC vs calibration set size.

Generates:
  1. Main figure (DeBERTa, MD only): 2 rows x 2 cols (MCE + ROCAUC)
  2. Appendix MCE+ROCAUC figures per score (MD, SP): 6 rows x 2 cols
  3. Appendix ECE figures per score (MD, SP): 6 rows x 1 col

Usage:
    python scripts/plot_ncal_ablation.py \
        --input results/ncal_ablation/ncal_ablation.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# --- Style ---
COLORS = {
    "um": "#1f77b4",        # blue
    "platt": "#ff7f0e",     # orange
    "isotonic": "#2ca02c",  # green
}
LABELS = {
    "um": "Uniform Mass",
    "platt": "Platt Scaling",
    "isotonic": "Isotonic Regression",
}
METHODS = ["um", "platt", "isotonic"]

DATASET_NAMES = {"sst2": "SST-2", "agnews": "AG News"}
MODEL_NAMES = {"electra": "ELECTRA", "bert": "BERT", "deberta": "DeBERTa"}
SCORE_NAMES = {"md": "Mahalanobis Distance", "sp": "Softmax Probability"}

# Metrics collected from the JSON
METRICS = ["mce", "rocauc", "ece"]


def compute_theoretical_epsilon(n_cal, alpha=0.05):
    """Compute UM theoretical guarantee epsilon."""
    B = int(2 * (n_cal ** (1 / 3)))
    if B < 1:
        return float("inf")
    n_per_bin = (n_cal + 1) // B - 1
    if n_per_bin < 1:
        return float("inf")
    return math.sqrt(math.log(2 * B / alpha) / (2 * n_per_bin))


def load_and_aggregate(json_path, model_filter=None, score_filter=None):
    """Load JSON results and aggregate by (method, n_cal, dataset).

    Returns:
        agg: dict (method, n_cal, dataset) -> {"mce": [...], "rocauc": [...], "ece": [...]}
        n_cal_values, raw_rocauc_means, alpha, datasets
    """
    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    n_cal_values = sorted(data["config"]["n_cal_values"])

    agg = {}
    raw_rocaucs = {}

    for r in results:
        if model_filter and r["model"] != model_filter:
            continue
        r_score = r.get("score", "md")
        if score_filter and r_score != score_filter:
            continue
        ds = r["dataset"]
        key = (r["method"], r["n_cal"], ds)
        if key not in agg:
            agg[key] = {m: [] for m in METRICS}
        for m in METRICS:
            if m in r and not math.isnan(r[m]):
                agg[key][m].append(r[m])
        if r["method"] == "um":
            raw_rocaucs.setdefault(ds, []).append(r["raw_rocauc"])

    datasets = sorted(raw_rocaucs.keys())
    raw_rocauc_means = {ds: np.mean(vals) for ds, vals in raw_rocaucs.items()}
    alpha = data["config"].get("alpha", 0.05)

    return agg, n_cal_values, raw_rocauc_means, alpha, datasets


def has_score(json_path, score_name):
    """Check if a score exists in the JSON data."""
    with open(json_path) as f:
        data = json.load(f)
    for r in data["results"]:
        if r.get("score", "md") == score_name:
            return True
    return False


def has_metric(json_path, metric_name):
    """Check if a metric exists in the JSON data."""
    with open(json_path) as f:
        data = json.load(f)
    return metric_name in data["results"][0]


def _plot_metric_on_ax(ax, agg, dataset, n_cal_values, metric, alpha,
                       show_epsilon=False, show_raw_rocauc=False,
                       raw_rocauc=None, show_legend=False, show_xlabel=True):
    """Plot a single metric on one axes for one dataset."""
    # Theoretical epsilon curve (only for MCE)
    if show_epsilon:
        n_smooth = np.logspace(
            np.log10(min(n_cal_values)), np.log10(max(n_cal_values)), 200
        )
        eps_smooth = [compute_theoretical_epsilon(int(n), alpha) for n in n_smooth]
        ax.plot(n_smooth, eps_smooth, color="black", linestyle="--",
                linewidth=1.2, label=r"Theor. $\varepsilon$", zorder=5)

    for method in METHODS:
        means, stds = [], []
        valid_ncals = []
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


def plot_row(ax_mce, ax_roc, agg, dataset, n_cal_values, raw_rocauc, alpha,
             show_legend=False, show_xlabel=True):
    """Plot one row (MCE + ROCAUC) for a single dataset."""
    _plot_metric_on_ax(ax_mce, agg, dataset, n_cal_values, "mce", alpha,
                       show_epsilon=True, show_legend=show_legend,
                       show_xlabel=show_xlabel)
    _plot_metric_on_ax(ax_roc, agg, dataset, n_cal_values, "rocauc", alpha,
                       show_raw_rocauc=True, raw_rocauc=raw_rocauc,
                       show_legend=show_legend, show_xlabel=show_xlabel)


# ---------- Main figure: DeBERTa, MD, 2x2 ----------

def plot_main_figure(json_path, output_path):
    """Generate main figure: DeBERTa only, MD score, 2 rows x 2 cols."""
    agg, n_cal_values, raw_rocauc_means, alpha, datasets = load_and_aggregate(
        json_path, model_filter="deberta", score_filter="md"
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    dataset_order = ["sst2", "agnews"]

    for row_idx, ds in enumerate(dataset_order):
        ds_name = DATASET_NAMES[ds]
        raw_roc = raw_rocauc_means.get(ds, 0.0)
        is_last = (row_idx == len(dataset_order) - 1)
        plot_row(
            axes[row_idx, 0], axes[row_idx, 1],
            agg, ds, n_cal_values, raw_roc, alpha,
            show_legend=(row_idx == 0),
            show_xlabel=is_last,
        )
        axes[row_idx, 0].set_title(f"{ds_name} — MCE")
        axes[row_idx, 1].set_title(f"{ds_name} — ROCAUC")
        axes[row_idx, 0].set_ylabel("MCE")
        axes[row_idx, 1].set_ylabel("ROCAUC")

    for ax in axes[-1]:
        ax.set_xlabel(r"$n_{\mathrm{cal}}$")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved main figure to {output_path}")
    plt.close(fig)


# ---------- Appendix: MCE+ROCAUC, 6x2 ----------

def _set_compact_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })


def plot_appendix_mce_rocauc(json_path, output_path, score_filter="md"):
    """Appendix figure: MCE + ROCAUC, all models, 6 rows x 2 cols."""
    model_order = ["electra", "bert", "deberta"]
    dataset_order = ["sst2", "agnews"]

    model_aggs = {}
    for model in model_order:
        model_agg, n_cal_values, model_raw, alpha, _ = load_and_aggregate(
            json_path, model_filter=model, score_filter=score_filter
        )
        model_aggs[model] = (model_agg, model_raw)

    _, n_cal_values, _, alpha, _ = load_and_aggregate(
        json_path, score_filter=score_filter
    )

    _set_compact_style()

    n_rows = len(model_order) * len(dataset_order)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(6.5, 1.4 * n_rows + 0.5),
    )
    fig.subplots_adjust(
        left=0.13, right=0.97, top=0.94, bottom=0.07,
        hspace=0.25, wspace=0.22,
    )

    row_idx = 0
    for model in model_order:
        model_agg, model_raw = model_aggs[model]
        for ds in dataset_order:
            raw_roc = model_raw.get(ds, 0.0)
            is_last = (row_idx == n_rows - 1)
            is_first = (row_idx == 0)

            plot_row(
                axes[row_idx, 0], axes[row_idx, 1],
                model_agg, ds, n_cal_values, raw_roc, alpha,
                show_legend=is_first, show_xlabel=is_last,
            )

            row_label = f"{MODEL_NAMES[model]} — {DATASET_NAMES[ds]}"
            axes[row_idx, 0].set_ylabel(row_label, fontsize=7.5,
                                         fontweight="bold")
            axes[row_idx, 1].set_ylabel("")
            row_idx += 1

    axes[0, 0].set_title("MCE", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("ROCAUC", fontsize=9, fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel(r"$n_{\mathrm{cal}}$", fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved MCE+ROCAUC appendix ({score_filter}) to {output_path}")
    plt.close(fig)


# ---------- Appendix: ECE only, 6x1 ----------

def plot_appendix_ece(json_path, output_path, score_filter="md"):
    """Appendix figure: ECE only, all models, 6 rows x 1 col."""
    model_order = ["electra", "bert", "deberta"]
    dataset_order = ["sst2", "agnews"]

    model_aggs = {}
    for model in model_order:
        model_agg, n_cal_values, _, alpha, _ = load_and_aggregate(
            json_path, model_filter=model, score_filter=score_filter
        )
        model_aggs[model] = model_agg

    _, n_cal_values, _, alpha, _ = load_and_aggregate(
        json_path, score_filter=score_filter
    )

    _set_compact_style()

    n_rows = len(model_order) * len(dataset_order)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(4.5, 1.4 * n_rows + 0.5),
    )
    fig.subplots_adjust(
        left=0.18, right=0.97, top=0.94, bottom=0.08,
        hspace=0.25,
    )

    row_idx = 0
    for model in model_order:
        model_agg = model_aggs[model]
        for ds in dataset_order:
            ax = axes[row_idx]
            is_last = (row_idx == n_rows - 1)
            is_first = (row_idx == 0)

            _plot_metric_on_ax(
                ax, model_agg, ds, n_cal_values, "ece", alpha,
                show_epsilon=False, show_legend=is_first,
                show_xlabel=is_last,
            )

            row_label = f"{MODEL_NAMES[model]} — {DATASET_NAMES[ds]}"
            ax.set_ylabel(row_label, fontsize=7.5, fontweight="bold")
            row_idx += 1

    axes[0].set_title("ECE", fontsize=9, fontweight="bold")
    axes[-1].set_xlabel(r"$n_{\mathrm{cal}}$", fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved ECE appendix ({score_filter}) to {output_path}")
    plt.close(fig)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Plot n_cal ablation results (main + appendix figures)"
    )
    parser.add_argument(
        "--input", type=str, default="results/ncal_ablation/ncal_ablation.json",
    )
    args = parser.parse_args()
    json_path = args.input
    fig_dir = "docs/paper/figures"

    print(f"Loading results from {json_path}")

    # Main figure: DeBERTa, MD
    plot_main_figure(json_path, f"{fig_dir}/ncal_ablation.pdf")

    # Appendix MCE+ROCAUC: MD
    plot_appendix_mce_rocauc(json_path, f"{fig_dir}/ncal_ablation_full.pdf",
                             score_filter="md")

    # Appendix MCE+ROCAUC: SP
    if has_score(json_path, "sp"):
        plot_appendix_mce_rocauc(json_path, f"{fig_dir}/ncal_ablation_sp.pdf",
                                 score_filter="sp")
    else:
        print("SP score not found — skipping")

    # Appendix ECE (only if data has ECE)
    if has_metric(json_path, "ece"):
        plot_appendix_ece(json_path, f"{fig_dir}/ncal_ablation_ece_md.pdf",
                          score_filter="md")
        if has_score(json_path, "sp"):
            plot_appendix_ece(json_path, f"{fig_dir}/ncal_ablation_ece_sp.pdf",
                              score_filter="sp")
    else:
        print("ECE metric not found — skipping ECE figures")


if __name__ == "__main__":
    main()
