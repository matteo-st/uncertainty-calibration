#!/usr/bin/env python3
"""
Plot n_cal ablation results: MCE and ROCAUC vs calibration set size.

Generates a 2-panel figure:
  Left:  MCE vs n_cal (log scale) with theoretical epsilon curve
  Right: ROCAUC vs n_cal (log scale) with raw MD baseline

Aggregates over (2 datasets x 3 models x 3 seeds x 20 draws) = 360 points
per n_cal value, reporting mean +/- std as line + shaded band.

Usage:
    python scripts/plot_ncal_ablation.py \
        --input results/ncal_ablation/ncal_ablation.json \
        --output docs/paper/figures/ncal_ablation.pdf
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


def compute_theoretical_epsilon(n_cal, alpha=0.05):
    """Compute UM theoretical guarantee epsilon."""
    B = int(2 * (n_cal ** (1 / 3)))
    if B < 1:
        return float("inf")
    n_per_bin = (n_cal + 1) // B - 1
    if n_per_bin < 1:
        return float("inf")
    return math.sqrt(math.log(2 * B / alpha) / (2 * n_per_bin))


def load_and_aggregate(json_path):
    """Load JSON results and aggregate by (method, n_cal)."""
    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    n_cal_values = sorted(data["config"]["n_cal_values"])

    # Aggregate per (method, n_cal)
    agg = {}  # (method, n_cal) -> {"mce": [...], "rocauc": [...]}
    raw_rocaucs = []

    for r in results:
        key = (r["method"], r["n_cal"])
        if key not in agg:
            agg[key] = {"mce": [], "rocauc": []}
        if not math.isnan(r["mce"]):
            agg[key]["mce"].append(r["mce"])
        if not math.isnan(r["rocauc"]):
            agg[key]["rocauc"].append(r["rocauc"])
        if r["method"] == "um":  # collect raw_rocauc once per entry
            raw_rocaucs.append(r["raw_rocauc"])

    raw_rocauc_mean = np.mean(raw_rocaucs) if raw_rocaucs else 0.0

    return agg, n_cal_values, raw_rocauc_mean, data["config"].get("alpha", 0.05)


def plot_ablation(agg, n_cal_values, raw_rocauc_mean, alpha, output_path):
    """Generate the 2-panel figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, (ax_mce, ax_roc) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Theoretical epsilon curve (smooth) ---
    n_cal_smooth = np.logspace(
        np.log10(min(n_cal_values)), np.log10(max(n_cal_values)), 200
    )
    eps_smooth = [compute_theoretical_epsilon(int(n), alpha) for n in n_cal_smooth]

    ax_mce.plot(
        n_cal_smooth, eps_smooth,
        color="black", linestyle="--", linewidth=1.5,
        label=r"Theoretical $\varepsilon$", zorder=5,
    )

    # --- Plot each method ---
    for method in METHODS:
        mce_means, mce_stds = [], []
        roc_means, roc_stds = [], []
        valid_ncals = []

        for n_cal in n_cal_values:
            key = (method, n_cal)
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

        # MCE panel
        ax_mce.plot(valid_ncals, mce_means, color=color, marker="o",
                    markersize=4, linewidth=1.5, label=label, zorder=3)
        ax_mce.fill_between(valid_ncals,
                            mce_means - mce_stds, mce_means + mce_stds,
                            color=color, alpha=0.15, zorder=2)

        # ROCAUC panel
        ax_roc.plot(valid_ncals, roc_means, color=color, marker="o",
                    markersize=4, linewidth=1.5, label=label, zorder=3)
        ax_roc.fill_between(valid_ncals,
                            roc_means - roc_stds, roc_means + roc_stds,
                            color=color, alpha=0.15, zorder=2)

    # --- Raw ROCAUC baseline ---
    ax_roc.axhline(
        raw_rocauc_mean, color="gray", linestyle="--", linewidth=1.2,
        label="Raw MD (no calibration)", zorder=1,
    )

    # --- Formatting ---
    ax_mce.set_xscale("log")
    ax_mce.set_xlabel(r"$n_{\mathrm{cal}}$")
    ax_mce.set_ylabel("MCE")
    ax_mce.set_title("(a) Maximum Calibration Error")
    ax_mce.legend(loc="upper right")
    ax_mce.xaxis.set_major_formatter(ScalarFormatter())
    ax_mce.set_xticks(n_cal_values)
    ax_mce.set_xticklabels([str(v) for v in n_cal_values], rotation=45, ha="right")
    ax_mce.grid(True, alpha=0.3)

    ax_roc.set_xscale("log")
    ax_roc.set_xlabel(r"$n_{\mathrm{cal}}$")
    ax_roc.set_ylabel("ROCAUC")
    ax_roc.set_title("(b) Discrimination (ROCAUC)")
    ax_roc.legend(loc="lower right")
    ax_roc.xaxis.set_major_formatter(ScalarFormatter())
    ax_roc.set_xticks(n_cal_values)
    ax_roc.set_xticklabels([str(v) for v in n_cal_values], rotation=45, ha="right")
    ax_roc.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot n_cal ablation results"
    )
    parser.add_argument(
        "--input", type=str, default="results/ncal_ablation/ncal_ablation.json",
        help="Path to ablation JSON results",
    )
    parser.add_argument(
        "--output", type=str,
        default="docs/paper/figures/ncal_ablation.pdf",
        help="Output figure path",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.input}")
    agg, n_cal_values, raw_rocauc_mean, alpha = load_and_aggregate(args.input)

    print(f"  n_cal values: {n_cal_values}")
    print(f"  Raw MD ROCAUC mean: {raw_rocauc_mean:.4f}")
    print(f"  Alpha: {alpha}")

    plot_ablation(agg, n_cal_values, raw_rocauc_mean, alpha, args.output)


if __name__ == "__main__":
    main()
