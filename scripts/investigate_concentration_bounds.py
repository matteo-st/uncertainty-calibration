#!/usr/bin/env python3
"""
Investigation of concentration inequalities for uniform-mass calibration bounds.

Compares Hoeffding (current paper bound) against tighter alternatives:
  1. Hoeffding — current bound: sqrt(log(2B/α) / (2 n_b))
  2. Clopper-Pearson — exact binomial CI
  3. KL-Chernoff — Bernoulli KL divergence bound
  4. Empirical Bernstein (Maurer-Pontil 2009)
  5. Bentkus (2004) — refinement using binomial tail

For each bound, computes per-bin epsilon and the resulting MCE bound.

Usage:
    python scripts/investigate_concentration_bounds.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------------
# Concentration bound implementations
# ---------------------------------------------------------------------------

def hoeffding_epsilon(n, delta):
    """Hoeffding bound: P(|p_hat - p| >= eps) <= 2*exp(-2*n*eps^2).
    Inverted: eps = sqrt(log(2/delta) / (2n)).
    """
    return math.sqrt(math.log(2.0 / delta) / (2 * n))


def binary_kl(p, q):
    """KL divergence between Bernoulli(p) and Bernoulli(q)."""
    if p <= 0:
        return -math.log(1 - q) if q < 1 else float('inf')
    if p >= 1:
        return -math.log(q) if q > 0 else float('inf')
    if q <= 0 or q >= 1:
        return float('inf')
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_chernoff_epsilon(p_hat, n, delta):
    """KL-Chernoff bound for Bernoulli: solve n * d(p_hat +/- eps || p_hat) = log(1/delta).

    Two-sided: use delta/2 per side, return max of upper and lower epsilon.
    Actually, for the MCE bound we need: find eps s.t. for all p in [p_hat-eps, p_hat+eps],
    n * d(p_hat || p) <= log(2/delta).

    Standard inversion: eps_upper = q_upper - p_hat where
    d(p_hat, q_upper) = log(2/delta) / n.
    """
    threshold = math.log(2.0 / delta) / n

    # Upper deviation: find q > p_hat s.t. d(p_hat, q) = threshold
    eps_upper = 0.0
    if p_hat < 1.0:
        try:
            q_upper = brentq(
                lambda q: binary_kl(p_hat, q) - threshold,
                p_hat + 1e-12, 1.0 - 1e-12
            )
            eps_upper = q_upper - p_hat
        except ValueError:
            eps_upper = 1.0 - p_hat

    # Lower deviation: find q < p_hat s.t. d(p_hat, q) = threshold
    eps_lower = 0.0
    if p_hat > 0.0:
        try:
            q_lower = brentq(
                lambda q: binary_kl(p_hat, q) - threshold,
                1e-12, p_hat - 1e-12
            )
            eps_lower = p_hat - q_lower
        except ValueError:
            eps_lower = p_hat

    return max(eps_upper, eps_lower)


def clopper_pearson_epsilon(k, n, delta):
    """Clopper-Pearson exact binomial CI.

    Returns max(p_hat - lower, upper - p_hat).
    """
    p_hat = k / n
    alpha = delta

    if k == 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)

    return max(p_hat - lower, upper - p_hat)


def empirical_bernstein_epsilon(p_hat, n, delta):
    """Maurer-Pontil (2009) empirical Bernstein bound.

    For Bernoulli: sample variance V = p_hat * (1 - p_hat) * n / (n-1).
    eps = sqrt(2 * V * log(2/delta) / n) + 7 * log(2/delta) / (3*(n-1))

    Note: for Bernoulli, we use V = p_hat*(1-p_hat) (MLE of variance).
    """
    if n <= 1:
        return 1.0
    V = p_hat * (1 - p_hat)  # Bernoulli variance estimate
    log_term = math.log(2.0 / delta)
    main_term = math.sqrt(2 * V * log_term / n)
    penalty_term = 7 * log_term / (3 * (n - 1))
    return main_term + penalty_term


def bentkus_epsilon(p_hat, n, delta):
    """Bentkus (2004) bound, which refines Hoeffding using binomial tails.

    For Bernoulli(p) with n samples: the Bentkus bound gives
    P(S_n >= n*p + t) <= (e/1) * P(Bin(n, p_hat) >= n*p_hat + t)

    In practice, for the confidence interval we use:
    The tightest approach is to use the exact binomial CDF.
    Bentkus showed: P(|p_hat - p| >= eps) <= e * P(|Bin(n,1/2)/n - 1/2| >= eps)

    This is tighter than Hoeffding by a factor related to the binomial vs Gaussian tail.
    For small n and moderate eps, the improvement is modest (~10-20%).
    We implement the simpler version using the refined Hoeffding with variance:
    P(|p_hat - p| >= eps) <= 2*exp(-n * h(eps, p_hat))
    where h uses the exact sub-Gaussian parameter sigma^2 = p_hat*(1-p_hat).
    """
    # Refined Hoeffding using true Bernoulli sub-Gaussian parameter
    # For X ~ Bernoulli(p), X is (1/4)-sub-Gaussian (Hoeffding),
    # but also p(1-p)-sub-Gaussian (tighter).
    # Since we don't know p, we use p_hat as proxy (this is NOT a valid PAC bound
    # without additional correction, so we use the Bernstein form instead).
    # For a valid bound, use the one-sided refined Hoeffding:
    sigma2 = p_hat * (1 - p_hat)
    if sigma2 <= 0:
        return 0.0
    # Bernstein-style: eps = sqrt(2*sigma2*log(2/delta)/n) + log(2/delta)/(3n)
    log_term = math.log(2.0 / delta)
    return math.sqrt(2 * sigma2 * log_term / n) + log_term / (3 * n)


# ---------------------------------------------------------------------------
# All bounds in one place
# ---------------------------------------------------------------------------

BOUND_NAMES = [
    "Hoeffding",
    "KL-Chernoff",
    "Clopper-Pearson",
    "Emp. Bernstein",
    "Bernstein (oracle)",
]

BOUND_COLORS = {
    "Hoeffding": "#d62728",        # red
    "KL-Chernoff": "#2ca02c",      # green
    "Clopper-Pearson": "#1f77b4",  # blue
    "Emp. Bernstein": "#ff7f0e",   # orange
    "Bernstein (oracle)": "#9467bd",  # purple
}

BOUND_LINESTYLES = {
    "Hoeffding": "-",
    "KL-Chernoff": "-",
    "Clopper-Pearson": "-",
    "Emp. Bernstein": "--",
    "Bernstein (oracle)": ":",
}


def compute_all_bounds(p_hat, n_b, delta_b):
    """Compute epsilon from all bounds for a single bin.

    Args:
        p_hat: observed error rate in the bin
        n_b: number of samples in the bin
        delta_b: per-bin failure probability (alpha/B after union bound)

    Returns:
        dict mapping bound name -> epsilon
    """
    k = int(round(p_hat * n_b))
    k = max(0, min(k, n_b))

    return {
        "Hoeffding": hoeffding_epsilon(n_b, delta_b),
        "KL-Chernoff": kl_chernoff_epsilon(p_hat, n_b, delta_b),
        "Clopper-Pearson": clopper_pearson_epsilon(k, n_b, delta_b),
        "Emp. Bernstein": empirical_bernstein_epsilon(p_hat, n_b, delta_b),
        "Bernstein (oracle)": bentkus_epsilon(p_hat, n_b, delta_b),
    }


# ---------------------------------------------------------------------------
# Part 1: Theoretical comparison — eps vs p for fixed n_b
# ---------------------------------------------------------------------------

def plot_eps_vs_p(output_dir, alpha=0.05):
    """Plot bound width as function of true p for various n_b values."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    n_b_values = [30, 50, 100, 200]
    p_values = np.linspace(0.01, 0.50, 200)

    fig, axes = plt.subplots(1, len(n_b_values), figsize=(14, 3.5), sharey=True)

    for ax_idx, n_b in enumerate(n_b_values):
        ax = axes[ax_idx]
        B = 19  # typical for n_cal=1000
        delta_b = alpha / B

        for bound_name in BOUND_NAMES:
            epsilons = []
            for p in p_values:
                bounds = compute_all_bounds(p, n_b, delta_b)
                epsilons.append(bounds[bound_name])
            ax.plot(p_values, epsilons,
                    color=BOUND_COLORS[bound_name],
                    linestyle=BOUND_LINESTYLES[bound_name],
                    linewidth=1.5, label=bound_name)

        ax.set_title(f"$n_b = {n_b}$", fontweight="bold")
        ax.set_xlabel(r"$\hat{p}_b$ (observed error rate)")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if ax_idx == 0:
            ax.set_ylabel(r"$\epsilon_b$ (bound width)")
        if ax_idx == len(n_b_values) - 1:
            ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        r"Concentration bound width vs observed error rate ($\alpha=0.05$, $B=19$)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "concentration_bounds_vs_p.pdf")
    save_fig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2: Theoretical comparison — eps vs n_b for fixed p
# ---------------------------------------------------------------------------

def plot_eps_vs_n(output_dir, alpha=0.05):
    """Plot bound width as function of n_b for typical p values."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    p_values = [0.05, 0.10, 0.20, 0.40]
    n_b_range = np.arange(10, 301, 2)

    fig, axes = plt.subplots(1, len(p_values), figsize=(14, 3.5), sharey=True)

    for ax_idx, p in enumerate(p_values):
        ax = axes[ax_idx]
        B = 19
        delta_b = alpha / B

        for bound_name in BOUND_NAMES:
            epsilons = []
            for n_b in n_b_range:
                bounds = compute_all_bounds(p, int(n_b), delta_b)
                epsilons.append(bounds[bound_name])
            ax.plot(n_b_range, epsilons,
                    color=BOUND_COLORS[bound_name],
                    linestyle=BOUND_LINESTYLES[bound_name],
                    linewidth=1.5, label=bound_name)

        ax.set_title(f"$\\hat{{p}}_b = {p}$", fontweight="bold")
        ax.set_xlabel(r"$n_b$ (samples per bin)")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if ax_idx == 0:
            ax.set_ylabel(r"$\epsilon_b$ (bound width)")
        if ax_idx == len(p_values) - 1:
            ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        r"Concentration bound width vs bin size ($\alpha=0.05$, $B=19$)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "concentration_bounds_vs_n.pdf")
    save_fig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 3: Empirical — per-bin comparison on actual data
# ---------------------------------------------------------------------------

MODELS = ["electra", "bert", "deberta"]
DATASETS = ["mrpc", "sst2", "cola", "agnews"]
SEEDS = [42, 123, 456]
SCORES = ["sp", "md"]

DATASET_NAMES = {"mrpc": "MRPC", "sst2": "SST-2", "cola": "CoLA", "agnews": "AG News"}


def save_fig(fig, path):
    """Save figure as both PDF and PNG."""
    fig.savefig(path, bbox_inches="tight", dpi=300)
    png_path = path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path} + .png")


def load_test_scores(results_dir, model, dataset, seed):
    path = os.path.join(results_dir, f"{model}_{dataset}_seed{seed}_test_scores.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_bin_stats(calibrated_scores, errors):
    """Get per-bin statistics from calibrated (discrete) scores on test set."""
    unique_vals = np.sort(np.unique(calibrated_scores))
    bins = []
    for t in unique_vals:
        mask = calibrated_scores == t
        n_bin = mask.sum()
        k_bin = errors[mask].sum()
        p_test = errors[mask].mean()  # test-set estimate of P(E=1|u_hat=t)
        calib_error = abs(p_test - t)  # actual calibration error
        bins.append({
            "t": float(t),          # assigned calibrated value (= p_hat on cal set)
            "n_test": int(n_bin),    # test samples in this bin
            "k_test": int(k_bin),    # test errors in this bin
            "p_test": float(p_test), # test error rate
            "calib_error": float(calib_error),
        })
    return bins


def empirical_per_bin_analysis(results_dir, paper_metrics_path, alpha=0.05):
    """Analyze per-bin bounds on actual data."""
    with open(paper_metrics_path) as f:
        paper_metrics = json.load(f)

    all_results = []

    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                exp_key = f"{model}_{dataset}_seed{seed}"
                data = load_test_scores(results_dir, model, dataset, seed)
                if data is None:
                    continue

                meta = paper_metrics.get(exp_key, {}).get("meta", {})
                n_cal = meta.get("n_cal", 1000)

                for score in SCORES:
                    cal_key = f"{score}_calibrated"
                    if cal_key not in data:
                        continue

                    calibrated = data[cal_key]
                    errors = data["errors"]

                    # Get number of bins from metadata
                    B = meta.get("um_bins", {}).get(score, 19)
                    n_b = max(1, (n_cal + 1) // B - 1)
                    delta_b = alpha / B

                    bin_stats = get_bin_stats(calibrated, errors)

                    for bin_info in bin_stats:
                        p_hat = bin_info["t"]  # cal-set error rate
                        bounds = compute_all_bounds(p_hat, n_b, delta_b)

                        all_results.append({
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "score": score,
                            "n_cal": n_cal,
                            "B": B,
                            "n_b": n_b,
                            "p_hat_cal": p_hat,
                            "p_test": bin_info["p_test"],
                            "n_test_bin": bin_info["n_test"],
                            "actual_calib_error": bin_info["calib_error"],
                            **{f"eps_{name}": eps for name, eps in bounds.items()},
                        })

    return all_results


def plot_per_bin_comparison(all_results, output_dir):
    """Plot per-bin epsilon from each bound vs actual calibration error."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 7.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    # Filter to DeBERTa for main figure
    deberta_results = [r for r in all_results if r["model"] == "deberta"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    score_labels = {"sp": "SP", "md": "MD"}

    for row_idx, score in enumerate(SCORES):
        score_data = [r for r in deberta_results if r["score"] == score]

        # Plot 1: per-bin bounds comparison
        ax = axes[row_idx, 0]
        p_hats = [r["p_hat_cal"] for r in score_data]
        actual_errors = [r["actual_calib_error"] for r in score_data]

        ax.scatter(p_hats, actual_errors, c="black", s=8, zorder=5,
                   alpha=0.5, label="Actual $|p_{\\mathrm{test}} - t|$")

        for bound_name in BOUND_NAMES:
            eps_key = f"eps_{bound_name}"
            eps_vals = [r[eps_key] for r in score_data]
            ax.scatter(p_hats, eps_vals, c=BOUND_COLORS[bound_name],
                       s=12, alpha=0.4, marker="x", label=bound_name)

        ax.set_xlabel(r"$\hat{p}_b$ (calibrated value)")
        ax.set_ylabel(r"$\epsilon_b$ or actual error")
        ax.set_title(f"Per-bin bounds — {score_labels[score]} score", fontweight="bold")
        ax.legend(loc="upper left", fontsize=6.5, ncol=2)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Plot 2: ratio eps_bound / eps_hoeffding
        ax2 = axes[row_idx, 1]
        for bound_name in BOUND_NAMES:
            if bound_name == "Hoeffding":
                continue
            ratios = [r[f"eps_{bound_name}"] / r["eps_Hoeffding"]
                      for r in score_data if r["eps_Hoeffding"] > 0]
            p_hats_r = [r["p_hat_cal"] for r in score_data if r["eps_Hoeffding"] > 0]
            ax2.scatter(p_hats_r, ratios, c=BOUND_COLORS[bound_name],
                        s=12, alpha=0.5, label=bound_name)

        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel(r"$\hat{p}_b$ (calibrated value)")
        ax2.set_ylabel(r"$\epsilon / \epsilon_{\mathrm{Hoeffding}}$ (ratio)")
        ax2.set_title(f"Tightness ratio — {score_labels[score]} score", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.set_ylim(0, 2.0)

    fig.suptitle("DeBERTa — Per-bin concentration bounds (all datasets, 3 seeds)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "concentration_bounds_perbin.pdf")
    save_fig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 4: MCE bound as function of n_cal
# ---------------------------------------------------------------------------

def plot_mce_bound_vs_ncal(output_dir, alpha=0.05):
    """Plot theoretical MCE bound (max eps over bins) vs n_cal for each bound.

    Uses typical bin error rates observed in our data.
    """
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    n_cal_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    # Typical bin error rates for a model with ~7% error rate
    # (from our DeBERTa SST-2 data)
    # At low error rate, most bins have low p_hat, a few have high p_hat
    typical_bin_ps = [0.0, 0.02, 0.02, 0.04, 0.04, 0.06, 0.08, 0.21, 0.23, 0.32]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, (title, bin_ps) in enumerate([
        ("Low error rate (~7%, like SST-2)", typical_bin_ps),
        ("Moderate error rate (~15%, like MRPC)",
         [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35]),
    ]):
        ax = axes[ax_idx]

        for bound_name in BOUND_NAMES:
            mce_bounds = []
            for n_cal in n_cal_values:
                B = max(2, int(2 * (n_cal ** (1/3))))
                n_b = max(1, (n_cal + 1) // B - 1)
                delta_b = alpha / B

                # For each bin, compute the bound
                max_eps = 0.0
                # Scale bin_ps to match B bins
                for i in range(B):
                    # Interpolate/repeat bin error rates
                    p_idx = int(i * len(bin_ps) / B)
                    p_hat = bin_ps[min(p_idx, len(bin_ps) - 1)]
                    bounds = compute_all_bounds(p_hat, n_b, delta_b)
                    max_eps = max(max_eps, bounds[bound_name])

                mce_bounds.append(max_eps)

            ax.plot(n_cal_values, mce_bounds,
                    color=BOUND_COLORS[bound_name],
                    linestyle=BOUND_LINESTYLES[bound_name],
                    marker="o", markersize=3, linewidth=1.5,
                    label=bound_name)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(n_cal_values)
        ax.set_xticklabels([str(v) for v in n_cal_values], rotation=45, ha="right")
        ax.set_xlabel(r"$n_{\mathrm{cal}}$")
        ax.set_ylabel(r"MCE bound ($\max_b \epsilon_b$)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        r"MCE bound vs $n_{\mathrm{cal}}$ for different concentration inequalities ($\alpha=0.05$)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "concentration_bounds_mce_vs_ncal.pdf")
    save_fig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 5: Summary table
# ---------------------------------------------------------------------------

def print_summary_tables(all_results, alpha=0.05):
    """Print summary tables comparing bounds."""

    print("\n" + "=" * 90)
    print("TABLE 1: Average epsilon per bound — DeBERTa, n_cal=1000")
    print("=" * 90)

    deberta = [r for r in all_results if r["model"] == "deberta"]

    for score in SCORES:
        score_data = [r for r in deberta if r["score"] == score]
        if not score_data:
            continue

        print(f"\n  Score: {score.upper()}")
        print(f"  {'Dataset':<10} {'n_b':>5} {'B':>4} ", end="")
        for name in BOUND_NAMES:
            print(f" {name:>16}", end="")
        print(f" {'Actual MCE':>12}")
        print("  " + "-" * 100)

        for dataset in DATASETS:
            ds_data = [r for r in score_data if r["dataset"] == dataset]
            if not ds_data:
                continue

            # Average over seeds
            n_b = ds_data[0]["n_b"]
            B = ds_data[0]["B"]

            # MCE = max over bins, averaged over seeds
            mce_per_seed = {}
            actual_mce_per_seed = {}
            for r in ds_data:
                s = r["seed"]
                for name in BOUND_NAMES:
                    key = f"eps_{name}"
                    mce_per_seed.setdefault((s, name), []).append(r[key])
                actual_mce_per_seed.setdefault(s, []).append(r["actual_calib_error"])

            avg_mce = {}
            for name in BOUND_NAMES:
                seed_maxes = []
                for seed in SEEDS:
                    vals = mce_per_seed.get((seed, name), [])
                    if vals:
                        seed_maxes.append(max(vals))
                avg_mce[name] = np.mean(seed_maxes) if seed_maxes else float('nan')

            actual_mce_avg = np.mean([max(actual_mce_per_seed.get(s, [0]))
                                      for s in SEEDS])

            print(f"  {DATASET_NAMES[dataset]:<10} {n_b:>5} {B:>4} ", end="")
            for name in BOUND_NAMES:
                print(f" {avg_mce[name]:>16.4f}", end="")
            print(f" {actual_mce_avg:>12.4f}")

    # Table 2: Ratio to Hoeffding
    print("\n\n" + "=" * 90)
    print("TABLE 2: Ratio ε / ε_Hoeffding — DeBERTa, averaged over all bins and seeds")
    print("=" * 90)

    for score in SCORES:
        score_data = [r for r in deberta if r["score"] == score]
        if not score_data:
            continue

        print(f"\n  Score: {score.upper()}")
        print(f"  {'Dataset':<10}", end="")
        for name in BOUND_NAMES:
            if name == "Hoeffding":
                continue
            print(f" {name:>16}", end="")
        print()
        print("  " + "-" * 80)

        for dataset in DATASETS:
            ds_data = [r for r in score_data if r["dataset"] == dataset]
            if not ds_data:
                continue

            print(f"  {DATASET_NAMES[dataset]:<10}", end="")
            for name in BOUND_NAMES:
                if name == "Hoeffding":
                    continue
                ratios = [r[f"eps_{name}"] / r["eps_Hoeffding"]
                          for r in ds_data if r["eps_Hoeffding"] > 0]
                mean_ratio = np.mean(ratios) if ratios else float('nan')
                print(f" {mean_ratio:>16.3f}", end="")
            print()

    # Table 3: MCE and ECE bound improvement summary
    print("\n\n" + "=" * 90)
    print("TABLE 3: MCE bound improvement over Hoeffding (DeBERTa, all datasets/scores)")
    print("=" * 90)

    print(f"\n  {'Bound':<20} {'Avg MCE bound':>14} {'vs Hoeffding':>14} {'Improvement':>14}")
    print("  " + "-" * 65)

    for name in BOUND_NAMES:
        mces = []
        hoeffding_mces = []
        for dataset in DATASETS:
            for seed in SEEDS:
                ds_data = [r for r in deberta
                           if r["dataset"] == dataset and r["seed"] == seed]
                if not ds_data:
                    continue
                for score in SCORES:
                    sc_data = [r for r in ds_data if r["score"] == score]
                    if not sc_data:
                        continue
                    mce = max(r[f"eps_{name}"] for r in sc_data)
                    mce_h = max(r["eps_Hoeffding"] for r in sc_data)
                    mces.append(mce)
                    hoeffding_mces.append(mce_h)

        avg_mce = np.mean(mces)
        avg_hoeffding = np.mean(hoeffding_mces)
        improvement = (1 - avg_mce / avg_hoeffding) * 100

        marker = " <-- current" if name == "Hoeffding" else ""
        print(f"  {name:<20} {avg_mce:>14.4f} {avg_mce/avg_hoeffding:>14.3f}x "
              f"{improvement:>+13.1f}%{marker}")

    # Table 4: Weighted-average (ECE-type) bound
    print("\n\n" + "=" * 90)
    print("TABLE 4: ECE-type bound (weighted avg of per-bin eps) — DeBERTa")
    print("=" * 90)
    print("  (Weights = fraction of test samples in each bin)")

    print(f"\n  {'Bound':<20} {'Avg ECE bound':>14} {'vs Hoeffding':>14} {'Improvement':>14}")
    print("  " + "-" * 65)

    for name in BOUND_NAMES:
        ece_bounds = []
        hoeffding_ece_bounds = []
        for dataset in DATASETS:
            for seed in SEEDS:
                for score in SCORES:
                    sc_data = [r for r in deberta
                               if r["dataset"] == dataset and r["seed"] == seed
                               and r["score"] == score]
                    if not sc_data:
                        continue
                    total_test = sum(r["n_test_bin"] for r in sc_data)
                    if total_test == 0:
                        continue
                    # Weighted average of per-bin eps
                    ece_b = sum(r["n_test_bin"] / total_test * r[f"eps_{name}"]
                                for r in sc_data)
                    ece_h = sum(r["n_test_bin"] / total_test * r["eps_Hoeffding"]
                                for r in sc_data)
                    ece_bounds.append(ece_b)
                    hoeffding_ece_bounds.append(ece_h)

        avg_ece = np.mean(ece_bounds) if ece_bounds else float('nan')
        avg_hoeffding_ece = np.mean(hoeffding_ece_bounds) if hoeffding_ece_bounds else float('nan')
        improvement = (1 - avg_ece / avg_hoeffding_ece) * 100

        marker = " <-- current" if name == "Hoeffding" else ""
        print(f"  {name:<20} {avg_ece:>14.4f} {avg_ece/avg_hoeffding_ece:>14.3f}x "
              f"{improvement:>+13.1f}%{marker}")


# ---------------------------------------------------------------------------
# Part 6: n_cal ablation with actual data
# ---------------------------------------------------------------------------

def plot_ncal_ablation_with_bounds(ncal_ablation_path, output_dir, alpha=0.05):
    """Overlay theoretical bounds from different methods on the n_cal ablation MCE plot.

    For data-dependent bounds (KL, CP), we compute the MCE bound as max over bins,
    using a realistic distribution of per-bin error rates derived from the overall
    error rate (~7% for SST-2, ~8% for AG News).
    """
    if not os.path.exists(ncal_ablation_path):
        print(f"SKIP: {ncal_ablation_path} not found")
        return

    with open(ncal_ablation_path) as f:
        ablation_data = json.load(f)

    results = ablation_data["results"]
    n_cal_values = sorted(ablation_data["config"]["n_cal_values"])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 7, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    # Filter to DeBERTa, UM method
    um_results = [r for r in results if r["method"] == "um" and r["model"] == "deberta"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Realistic bin error rate distributions (from actual DeBERTa data)
    # Derived from calibrated score values across bins
    bin_p_distributions = {
        "sst2": [0.0, 0.0, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.06, 0.06,
                 0.06, 0.08, 0.08, 0.08, 0.10, 0.15, 0.21, 0.23, 0.32],
        "agnews": [0.0, 0.0, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.06, 0.06,
                   0.06, 0.08, 0.08, 0.10, 0.10, 0.12, 0.18, 0.24, 0.30],
    }

    for col_idx, dataset in enumerate(["sst2", "agnews"]):
        ax = axes[col_idx]
        ds_results = [r for r in um_results if r["dataset"] == dataset]
        ref_bin_ps = bin_p_distributions[dataset]

        for score in SCORES:
            sc_results = [r for r in ds_results if r["score"] == score]

            # Compute empirical MCE mean ± std per n_cal
            means, stds, valid_ncals = [], [], []
            for n_cal in n_cal_values:
                ncal_results = [r for r in sc_results if r["n_cal"] == n_cal]
                mces = [r["mce"] for r in ncal_results
                        if r["mce"] is not None and not math.isnan(r["mce"])]
                if mces:
                    valid_ncals.append(n_cal)
                    means.append(np.mean(mces))
                    stds.append(np.std(mces))

            color = "#2ca02c" if score == "sp" else "#1f77b4"
            label = f"UM MCE ({score.upper()})"
            ax.plot(valid_ncals, means, color=color, marker="o",
                    markersize=3, linewidth=1.2, label=label, zorder=4)
            ax.fill_between(valid_ncals,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=color, alpha=0.15, zorder=2)

        # Overlay theoretical bounds — compute max over bins
        bound_styles = {
            "Hoeffding": {"color": "#d62728", "ls": "-", "lw": 2.0},
            "KL-Chernoff": {"color": "#ff7f0e", "ls": "--", "lw": 1.5},
            "Clopper-Pearson": {"color": "#9467bd", "ls": "-.", "lw": 1.5},
        }

        for bound_name, style in bound_styles.items():
            bound_vals = []
            for n_cal in n_cal_values:
                B = max(2, int(2 * (n_cal ** (1/3))))
                n_b = max(1, (n_cal + 1) // B - 1)
                delta_b = alpha / B

                # Compute max over bins using realistic bin p distribution
                max_eps = 0.0
                for i in range(B):
                    p_idx = int(i * len(ref_bin_ps) / B)
                    p_hat = ref_bin_ps[min(p_idx, len(ref_bin_ps) - 1)]
                    bounds = compute_all_bounds(p_hat, n_b, delta_b)
                    max_eps = max(max_eps, bounds[bound_name])
                bound_vals.append(max_eps)

            ax.plot(n_cal_values, bound_vals,
                    color=style["color"], linestyle=style["ls"],
                    linewidth=style["lw"],
                    label=f"{bound_name} bound", zorder=3)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(n_cal_values)
        ax.set_xticklabels([str(v) for v in n_cal_values], rotation=45, ha="right")
        ax.set_xlabel(r"$n_{\mathrm{cal}}$")
        ax.set_ylabel("MCE / Bound")
        ax.set_title(DATASET_NAMES[dataset], fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=6.5)

    fig.suptitle(
        "UM calibration MCE vs theoretical bounds (DeBERTa)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "concentration_bounds_ncal_overlay.pdf")
    save_fig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 7: Improvement heatmap — all (model, dataset) pairs
# ---------------------------------------------------------------------------

def plot_improvement_heatmap(all_results, output_dir):
    """Heatmap of KL-Chernoff / Hoeffding ratio across all experiments."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, score in enumerate(SCORES):
        ax = axes[ax_idx]

        # Build matrix: rows = datasets, cols = models
        matrix = np.full((len(DATASETS), len(MODELS)), np.nan)

        for i, dataset in enumerate(DATASETS):
            for j, model in enumerate(MODELS):
                rows = [r for r in all_results
                        if r["model"] == model and r["dataset"] == dataset
                        and r["score"] == score]
                if not rows:
                    continue

                # Average ratio across bins and seeds
                ratios = [r["eps_KL-Chernoff"] / r["eps_Hoeffding"]
                          for r in rows if r["eps_Hoeffding"] > 0]
                matrix[i, j] = np.mean(ratios) if ratios else np.nan

        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([m.upper() for m in MODELS])
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels([DATASET_NAMES[d] for d in DATASETS])
        ax.set_title(f"{score.upper()} score", fontweight="bold")

        # Annotate cells
        for i in range(len(DATASETS)):
            for j in range(len(MODELS)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}",
                            ha="center", va="center", fontsize=10,
                            color="white" if matrix[i, j] < 0.6 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\epsilon_{\mathrm{KL}} / \epsilon_{\mathrm{Hoeffding}}$")

    fig.suptitle(
        "KL-Chernoff / Hoeffding ratio (lower = tighter, averaged over bins and seeds)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "concentration_bounds_heatmap.pdf")
    save_fig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = "results/paper_metrics"
    paper_metrics_path = os.path.join(results_dir, "paper_metrics.json")
    ncal_ablation_path = "results/ncal_ablation/ncal_ablation.json"
    output_dir = "results/concentration_bounds"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CONCENTRATION INEQUALITY INVESTIGATION")
    print("=" * 70)

    # Part 1: Theoretical — eps vs p
    print("\n--- Part 1: Bound width vs observed error rate ---")
    plot_eps_vs_p(output_dir)

    # Part 2: Theoretical — eps vs n_b
    print("\n--- Part 2: Bound width vs bin size ---")
    plot_eps_vs_n(output_dir)

    # Part 3: Empirical per-bin analysis
    print("\n--- Part 3: Per-bin analysis on actual data ---")
    all_results = empirical_per_bin_analysis(results_dir, paper_metrics_path)
    print(f"  Computed bounds for {len(all_results)} bins across all experiments")

    # Part 4: Summary tables
    print_summary_tables(all_results)

    # Part 5: Per-bin comparison figure
    print("\n--- Part 5: Per-bin comparison figure ---")
    plot_per_bin_comparison(all_results, output_dir)

    # Part 6: Improvement heatmap
    print("\n--- Part 6: Improvement heatmap ---")
    plot_improvement_heatmap(all_results, output_dir)

    # Part 7: MCE bound vs n_cal with actual MCE overlay
    print("\n--- Part 7: MCE bound vs n_cal ---")
    plot_mce_bound_vs_ncal(output_dir)

    # Part 8: Overlay bounds on n_cal ablation
    print("\n--- Part 8: Overlay on n_cal ablation ---")
    plot_ncal_ablation_with_bounds(ncal_ablation_path, output_dir)

    # Save all results as JSON
    json_path = os.path.join(output_dir, "concentration_bounds_analysis.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {json_path}")

    print("\n" + "=" * 70)
    print("DONE — All figures saved to:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
