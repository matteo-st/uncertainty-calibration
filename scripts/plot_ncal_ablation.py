#!/usr/bin/env python3
"""
Plot n_cal ablation results: MCE and ROCAUC vs calibration set size.

Generates:
  1. Main figure (DeBERTa, MD only): 2 rows x 2 cols
  2. Appendix figure per score (MD, SP): 6 rows x 2 cols, compact

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

    Args:
        json_path: Path to ablation JSON file.
        model_filter: If set, keep only this model (e.g. "deberta").
        score_filter: If set, keep only this score (e.g. "md").

    Returns:
        agg: dict (method, n_cal, dataset) -> {"mce": [...], "rocauc": [...]}
        n_cal_values: sorted list of n_cal values
        raw_rocaucs: dict dataset -> list of raw_rocauc values
        alpha: significance level
        datasets: sorted list of dataset keys present
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
        # Handle both old format (no "score" field = MD only) and new format
        r_score = r.get("score", "md")
        if score_filter and r_score != score_filter:
            continue
        ds = r["dataset"]
        key = (r["method"], r["n_cal"], ds)
        if key not in agg:
            agg[key] = {"mce": [], "rocauc": []}
        if not math.isnan(r["mce"]):
            agg[key]["mce"].append(r["mce"])
        if not math.isnan(r["rocauc"]):
            agg[key]["rocauc"].append(r["rocauc"])
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


def plot_row(ax_mce, ax_roc, agg, dataset, n_cal_values, raw_rocauc, alpha,
             show_legend=False, show_xlabel=True):
    """Plot one row (MCE + ROCAUC) for a single dataset."""
    # Theoretical epsilon curve
    n_cal_smooth = np.logspace(
        np.log10(min(n_cal_values)), np.log10(max(n_cal_values)), 200
    )
    eps_smooth = [compute_theoretical_epsilon(int(n), alpha) for n in n_cal_smooth]

    ax_mce.plot(
        n_cal_smooth, eps_smooth,
        color="black", linestyle="--", linewidth=1.2,
        label=r"Theor. $\varepsilon$", zorder=5,
    )

    for method in METHODS:
        mce_means, mce_stds = [], []
        roc_means, roc_stds = [], []
        valid_ncals = []

        for n_cal in n_cal_values:
            key = (method, n_cal, dataset)
            if key not in agg or not agg[key]["mce"]:
                continue
            valid_ncals.append(n_cal)
            mce_means.append(np.mean(agg[key]["mce"]))
            mce_stds.append(np.std(agg[key]["mce"]))
            roc_means.append(np.mean(agg[key]["rocauc"]))
            roc_stds.append(np.std(agg[key]["rocauc"]))

        valid_ncals = np.array(valid_ncals)
        mce_means = np.array(mce_means)
        mce_stds = np.array(mce_stds)
        roc_means = np.array(roc_means)
        roc_stds = np.array(roc_stds)

        color = COLORS[method]
        label = LABELS[method]

        ax_mce.plot(valid_ncals, mce_means, color=color, marker="o",
                    markersize=3, linewidth=1.2, label=label, zorder=3)
        ax_mce.fill_between(valid_ncals,
                            mce_means - mce_stds, mce_means + mce_stds,
                            color=color, alpha=0.15, zorder=2)

        ax_roc.plot(valid_ncals, roc_means, color=color, marker="o",
                    markersize=3, linewidth=1.2, label=label, zorder=3)
        ax_roc.fill_between(valid_ncals,
                            roc_means - roc_stds, roc_means + roc_stds,
                            color=color, alpha=0.15, zorder=2)

    ax_roc.axhline(
        raw_rocauc, color="gray", linestyle="--", linewidth=1.0,
        label="Raw (no cal.)", zorder=1,
    )

    for ax in [ax_mce, ax_roc]:
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
        ax_mce.legend(loc="upper right", fontsize=7, handlelength=1.5,
                      borderpad=0.3, labelspacing=0.3)
        ax_roc.legend(loc="lower right", fontsize=7, handlelength=1.5,
                      borderpad=0.3, labelspacing=0.3)


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

    # x-label only on bottom row
    for ax in axes[-1]:
        ax.set_xlabel(r"$n_{\mathrm{cal}}$")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved main figure to {output_path}")
    plt.close(fig)


# ---------- Appendix figure: all models, compact 6x2 ----------

def plot_appendix_figure(json_path, output_path, score_filter="md"):
    """Generate appendix figure: all models, 6 rows x 2 cols, compact."""
    model_order = ["electra", "bert", "deberta"]
    dataset_order = ["sst2", "agnews"]

    # Collect per-model aggregations
    model_aggs = {}
    for model in model_order:
        model_agg, n_cal_values, model_raw, alpha, _ = load_and_aggregate(
            json_path, model_filter=model, score_filter=score_filter
        )
        model_aggs[model] = (model_agg, model_raw)

    # Also get n_cal_values and alpha from first load
    _, n_cal_values, _, alpha, _ = load_and_aggregate(
        json_path, score_filter=score_filter
    )

    # Compact style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    n_rows = len(model_order) * len(dataset_order)  # 6
    row_h = 1.4
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(6.5, row_h * n_rows + 0.5),
    )
    fig.subplots_adjust(
        left=0.13, right=0.97, top=0.94, bottom=0.07,
        hspace=0.25, wspace=0.22,
    )

    row_idx = 0
    for model in model_order:
        model_agg, model_raw = model_aggs[model]
        for ds in dataset_order:
            ds_name = DATASET_NAMES[ds]
            model_name = MODEL_NAMES[model]
            raw_roc = model_raw.get(ds, 0.0)
            is_last_row = (row_idx == n_rows - 1)
            is_first_row = (row_idx == 0)

            plot_row(
                axes[row_idx, 0], axes[row_idx, 1],
                model_agg, ds, n_cal_values, raw_roc, alpha,
                show_legend=is_first_row,
                show_xlabel=is_last_row,
            )

            # Row label as the left subplot ylabel
            row_label = f"{model_name} — {ds_name}"
            axes[row_idx, 0].set_ylabel(row_label, fontsize=7.5,
                                         fontweight="bold")
            # No ylabel on right column
            axes[row_idx, 1].set_ylabel("")

            row_idx += 1

    # Column headers on first row only
    axes[0, 0].set_title("MCE", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("ROCAUC", fontsize=9, fontweight="bold")

    # x-label only on bottom
    for ax in axes[-1]:
        ax.set_xlabel(r"$n_{\mathrm{cal}}$", fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved appendix figure ({score_filter}) to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot n_cal ablation results (main + appendix figures)"
    )
    parser.add_argument(
        "--input", type=str, default="results/ncal_ablation/ncal_ablation.json",
        help="Path to ablation JSON results",
    )
    parser.add_argument(
        "--output-main", type=str,
        default="docs/paper/figures/ncal_ablation.pdf",
        help="Output path for main figure (DeBERTa, 2x2)",
    )
    parser.add_argument(
        "--output-appendix-md", type=str,
        default="docs/paper/figures/ncal_ablation_full.pdf",
        help="Output path for appendix figure — MD score",
    )
    parser.add_argument(
        "--output-appendix-sp", type=str,
        default="docs/paper/figures/ncal_ablation_sp.pdf",
        help="Output path for appendix figure — SP score",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.input}")

    # Main figure: DeBERTa, MD only
    plot_main_figure(args.input, args.output_main)

    # Appendix: MD (always available)
    plot_appendix_figure(args.input, args.output_appendix_md, score_filter="md")

    # Appendix: SP (only if data available)
    if has_score(args.input, "sp"):
        plot_appendix_figure(args.input, args.output_appendix_sp, score_filter="sp")
    else:
        print("SP score not found in data — skipping SP appendix figure")


if __name__ == "__main__":
    main()
