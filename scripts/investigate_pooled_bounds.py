#!/usr/bin/env python3
"""
Synthetic experiment: pooled (cal+test) calibration certificates.

Three certificate types for |p_b - t_b| in each UM bin:
  1. Cal-only:  ε(n_b^cal, δ_b)
  2. Test-only: |p̂_test,b - t_b| + ε(n_b^test, δ_b)
  3. Pooled:    (n_test/n_pool)|p̂_test,b - t_b| + ε(n_b^pool, δ_b)

Uses synthetic data with KNOWN true calibration to verify coverage
and compare tightness.

Usage:
    python scripts/investigate_pooled_bounds.py
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Concentration bounds
# ---------------------------------------------------------------------------

def hoeffding_epsilon(n, delta):
    return math.sqrt(math.log(2.0 / delta) / (2 * n))


def binary_kl(p, q):
    if p <= 0:
        return -math.log(1 - q) if q < 1 else float('inf')
    if p >= 1:
        return -math.log(q) if q > 0 else float('inf')
    if q <= 0 or q >= 1:
        return float('inf')
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_chernoff_epsilon(p_hat, n, delta):
    threshold = math.log(2.0 / delta) / n
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
    p_hat = k / n
    if k == 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(delta / 2, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1 - delta / 2, k + 1, n - k)
    return max(p_hat - lower, upper - p_hat)


def compute_eps(p_hat, n, k, delta, bound="cp"):
    """Compute epsilon for one group."""
    if bound == "hoeffding":
        return hoeffding_epsilon(n, delta)
    elif bound == "kl":
        return kl_chernoff_epsilon(p_hat, n, delta)
    elif bound == "cp":
        return clopper_pearson_epsilon(k, n, delta)
    raise ValueError(f"Unknown bound: {bound}")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def true_error_prob(score, scenario="moderate"):
    """True P(error | score) — known ground truth."""
    if scenario == "moderate":
        # Sigmoid-like: P(error) goes from ~0.02 at score=0 to ~0.45 at score=1
        return 0.02 + 0.43 / (1 + np.exp(-8 * (score - 0.5)))
    elif scenario == "low_error":
        # Model with ~5% overall error
        return 0.01 + 0.15 / (1 + np.exp(-10 * (score - 0.6)))
    elif scenario == "high_error":
        # Model with ~25% overall error
        return 0.05 + 0.50 / (1 + np.exp(-6 * (score - 0.5)))
    raise ValueError(f"Unknown scenario: {scenario}")


def generate_data(n, rng, scenario="moderate"):
    """Generate synthetic (score, error) pairs."""
    # Scores ~ Beta(2, 5) — skewed toward low scores (most predictions correct)
    scores = rng.beta(2, 5, size=n)
    probs = true_error_prob(scores, scenario)
    errors = rng.binomial(1, probs).astype(float)
    return scores, errors


# ---------------------------------------------------------------------------
# UM calibration
# ---------------------------------------------------------------------------

def fit_um(cal_scores, cal_errors):
    """Fit UM calibration, return bin edges and calibrated values."""
    n = len(cal_scores)
    B = max(2, int(2 * (n ** (1/3))))
    quantiles = np.linspace(0, 100, B + 1)
    bin_edges = np.percentile(cal_scores, quantiles)
    bin_edges = np.unique(bin_edges)
    actual_B = len(bin_edges) - 1

    t_values = np.zeros(actual_B)
    n_b_cal = np.zeros(actual_B, dtype=int)
    k_b_cal = np.zeros(actual_B, dtype=int)

    for i in range(actual_B):
        if i == actual_B - 1:
            mask = (cal_scores >= bin_edges[i]) & (cal_scores <= bin_edges[i + 1])
        else:
            mask = (cal_scores >= bin_edges[i]) & (cal_scores < bin_edges[i + 1])
        n_b_cal[i] = mask.sum()
        k_b_cal[i] = int(cal_errors[mask].sum())
        t_values[i] = cal_errors[mask].mean() if mask.sum() > 0 else cal_errors.mean()

    return bin_edges, t_values, n_b_cal, k_b_cal


def assign_bins(scores, bin_edges):
    """Assign scores to bins, return bin indices."""
    idx = np.digitize(scores, bin_edges[1:-1])
    return np.clip(idx, 0, len(bin_edges) - 2)


def true_bin_error_rate(bin_edges, scenario, n_mc=100000):
    """Compute true P(error | score in bin b) by Monte Carlo."""
    rng = np.random.RandomState(99999)
    scores = rng.beta(2, 5, size=n_mc)
    probs = true_error_prob(scores, scenario)
    actual_B = len(bin_edges) - 1
    p_true = np.zeros(actual_B)
    for i in range(actual_B):
        if i == actual_B - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        else:
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() > 0:
            p_true[i] = probs[mask].mean()
    return p_true


# ---------------------------------------------------------------------------
# Certificate computation
# ---------------------------------------------------------------------------

def compute_certificates(bin_edges, t_values, n_b_cal, k_b_cal,
                         test_scores, test_errors, alpha, bound="cp"):
    """Compute cal-only, test-only, and pooled certificates."""
    actual_B = len(t_values)
    delta_b = alpha / actual_B

    test_bin_idx = assign_bins(test_scores, bin_edges)

    results_per_bin = []
    for b in range(actual_B):
        test_mask = test_bin_idx == b
        n_b_test = int(test_mask.sum())
        k_b_test = int(test_errors[test_mask].sum())

        n_b_pool = n_b_cal[b] + n_b_test
        k_b_pool = k_b_cal[b] + k_b_test

        # Cal-only certificate: ε(n_cal, δ)
        if n_b_cal[b] > 0:
            p_hat_cal = k_b_cal[b] / n_b_cal[b]
            eps_cal = compute_eps(p_hat_cal, n_b_cal[b], k_b_cal[b], delta_b, bound)
            cert_cal = eps_cal
        else:
            cert_cal = 1.0

        # Test-only certificate: |p̂_test - t| + ε(n_test, δ)
        if n_b_test > 0:
            p_hat_test = k_b_test / n_b_test
            eps_test = compute_eps(p_hat_test, n_b_test, k_b_test, delta_b, bound)
            cert_test = abs(p_hat_test - t_values[b]) + eps_test
        else:
            cert_test = 1.0

        # Pooled certificate: (n_test/n_pool)|p̂_test - t| + ε(n_pool, δ)
        if n_b_pool > 0 and n_b_test > 0:
            p_hat_pool = k_b_pool / n_b_pool
            eps_pool = compute_eps(p_hat_pool, n_b_pool, k_b_pool, delta_b, bound)
            cert_pool = (n_b_test / n_b_pool) * abs(p_hat_test - t_values[b]) + eps_pool
        elif n_b_pool > 0:
            p_hat_pool = k_b_pool / n_b_pool
            eps_pool = compute_eps(p_hat_pool, n_b_pool, k_b_pool, delta_b, bound)
            cert_pool = eps_pool
        else:
            cert_pool = 1.0

        # Best = min of all three
        cert_best = min(cert_cal, cert_test, cert_pool)

        results_per_bin.append({
            "b": b, "t": t_values[b],
            "n_cal": n_b_cal[b], "n_test": n_b_test, "n_pool": n_b_pool,
            "cert_cal": cert_cal, "cert_test": cert_test,
            "cert_pool": cert_pool, "cert_best": cert_best,
        })

    # MCE certificates = max over bins
    mce_cal = max(r["cert_cal"] for r in results_per_bin)
    mce_test = max(r["cert_test"] for r in results_per_bin)
    mce_pool = max(r["cert_pool"] for r in results_per_bin)
    mce_best = max(r["cert_best"] for r in results_per_bin)

    return {
        "mce_cal": mce_cal, "mce_test": mce_test,
        "mce_pool": mce_pool, "mce_best": mce_best,
        "bins": results_per_bin,
    }


def compute_true_mce(bin_edges, t_values, scenario):
    """Compute true MCE from known calibration function."""
    p_true = true_bin_error_rate(bin_edges, scenario)
    return max(abs(p_true[b] - t_values[b]) for b in range(len(t_values)))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_cal, n_test, n_repeats, scenario, alpha, bound):
    """Run Monte Carlo experiment for one (n_cal, n_test) pair."""
    coverage = {"cal": 0, "test": 0, "pool": 0, "best": 0}
    cert_values = {"cal": [], "test": [], "pool": [], "best": []}
    true_mces = []

    for rep in range(n_repeats):
        rng = np.random.RandomState(rep)

        # Generate cal and test data
        cal_scores, cal_errors = generate_data(n_cal, rng, scenario)
        test_scores, test_errors = generate_data(n_test, rng, scenario)

        # Fit UM on cal
        bin_edges, t_values, n_b_cal, k_b_cal = fit_um(cal_scores, cal_errors)

        # True MCE
        true_mce = compute_true_mce(bin_edges, t_values, scenario)
        true_mces.append(true_mce)

        # Certificates
        certs = compute_certificates(
            bin_edges, t_values, n_b_cal, k_b_cal,
            test_scores, test_errors, alpha, bound
        )

        for key in ["cal", "test", "pool", "best"]:
            val = certs[f"mce_{key}"]
            cert_values[key].append(val)
            if val >= true_mce - 1e-10:  # small tolerance for numerics
                coverage[key] += 1

    coverage_rate = {k: v / n_repeats for k, v in coverage.items()}
    mean_cert = {k: np.mean(v) for k, v in cert_values.items()}
    std_cert = {k: np.std(v) for k, v in cert_values.items()}
    mean_true_mce = np.mean(true_mces)

    return {
        "n_cal": n_cal, "n_test": n_test,
        "coverage": coverage_rate,
        "mean_cert": mean_cert,
        "std_cert": std_cert,
        "mean_true_mce": mean_true_mce,
    }


def save_fig(fig, path_no_ext):
    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{path_no_ext}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {path_no_ext}.pdf/.png")


def main():
    output_dir = "results/pooled_bounds"
    os.makedirs(output_dir, exist_ok=True)

    alpha = 0.05
    bound = "cp"
    n_repeats = 2000
    scenario = "moderate"

    n_cal_values = [100, 500, 1000]
    n_test_values = [50, 100, 200, 500, 1000, 2000, 5000]

    all_results = []

    print(f"Synthetic pooled bounds experiment")
    print(f"  Scenario: {scenario}, Bound: {bound}, α={alpha}")
    print(f"  n_repeats: {n_repeats}")
    print(f"  n_cal: {n_cal_values}")
    print(f"  n_test: {n_test_values}\n")

    for n_cal in n_cal_values:
        for n_test in n_test_values:
            print(f"  n_cal={n_cal:>5}, n_test={n_test:>5} ...", end=" ", flush=True)
            result = run_experiment(n_cal, n_test, n_repeats, scenario, alpha, bound)
            all_results.append(result)
            cov = result["coverage"]
            mc = result["mean_cert"]
            tm = result["mean_true_mce"]
            print(f"cov: cal={cov['cal']:.3f} test={cov['test']:.3f} "
                  f"pool={cov['pool']:.3f} best={cov['best']:.3f} | "
                  f"cert: cal={mc['cal']:.3f} test={mc['test']:.3f} "
                  f"pool={mc['pool']:.3f} best={mc['best']:.3f} | "
                  f"true_mce={tm:.3f}")

    # =========================================================================
    # TABLE 1: Coverage (should be ≥ 1-α = 0.95)
    # =========================================================================
    print("\n" + "=" * 100)
    print(f"TABLE 1: Coverage (should be ≥ {1-alpha:.2f})")
    print("=" * 100)
    print(f"{'n_cal':>6} {'n_test':>6} │ {'Cal-only':>9} {'Test-only':>10} "
          f"{'Pooled':>8} {'Best':>8}")
    print("-" * 60)
    for r in all_results:
        c = r["coverage"]
        print(f"{r['n_cal']:>6} {r['n_test']:>6} │ {c['cal']:>9.3f} "
              f"{c['test']:>10.3f} {c['pool']:>8.3f} {c['best']:>8.3f}")

    # =========================================================================
    # TABLE 2: Mean certificate value (tightness — lower is better)
    # =========================================================================
    print("\n" + "=" * 110)
    print("TABLE 2: Mean MCE certificate (lower = tighter, but must maintain coverage)")
    print("=" * 110)
    print(f"{'n_cal':>6} {'n_test':>6} │ {'True MCE':>9} {'Cal-only':>9} "
          f"{'Test-only':>10} {'Pooled':>8} {'Best':>8} │ "
          f"{'pool/test':>10} {'best/cal':>9}")
    print("-" * 110)
    for r in all_results:
        mc = r["mean_cert"]
        tm = r["mean_true_mce"]
        pool_vs_test = mc["pool"] / mc["test"] if mc["test"] > 0 else float('inf')
        best_vs_cal = mc["best"] / mc["cal"] if mc["cal"] > 0 else float('inf')
        print(f"{r['n_cal']:>6} {r['n_test']:>6} │ {tm:>9.4f} {mc['cal']:>9.4f} "
              f"{mc['test']:>10.4f} {mc['pool']:>8.4f} {mc['best']:>8.4f} │ "
              f"{pool_vs_test:>10.3f} {best_vs_cal:>9.3f}")

    # =========================================================================
    # TABLE 3: Improvement of pooled over test-only and cal-only
    # =========================================================================
    print("\n" + "=" * 90)
    print("TABLE 3: Improvement ratios")
    print("  pool/test: pooled vs test-only (< 1 means pooled is tighter)")
    print("  pool/cal:  pooled vs cal-only  (< 1 means pooled is tighter)")
    print("  best/cal:  best-of-3 vs cal-only")
    print("=" * 90)
    print(f"{'n_cal':>6} {'n_test':>6} │ {'pool/test':>10} {'pool/cal':>10} "
          f"{'best/cal':>10}")
    print("-" * 50)
    for r in all_results:
        mc = r["mean_cert"]
        print(f"{r['n_cal']:>6} {r['n_test']:>6} │ "
              f"{mc['pool']/mc['test']:>10.3f} "
              f"{mc['pool']/mc['cal']:>10.3f} "
              f"{mc['best']/mc['cal']:>10.3f}")

    # =========================================================================
    # FIGURE 1: Certificate tightness vs n_test (one subplot per n_cal)
    # =========================================================================
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, len(n_cal_values), figsize=(5 * len(n_cal_values), 4.5))
    if len(n_cal_values) == 1:
        axes = [axes]

    colors = {"cal": "gray", "test": "#1f77b4", "pool": "#2ca02c", "best": "#d62728"}
    labels = {"cal": "Cal-only", "test": "Test-only", "pool": "Pooled", "best": "Best of 3"}

    for ax_idx, n_cal in enumerate(n_cal_values):
        ax = axes[ax_idx]
        results_nc = [r for r in all_results if r["n_cal"] == n_cal]
        n_tests = [r["n_test"] for r in results_nc]

        for key in ["cal", "test", "pool", "best"]:
            means = [r["mean_cert"][key] for r in results_nc]
            stds = [r["std_cert"][key] for r in results_nc]
            means = np.array(means)
            stds = np.array(stds)
            ls = "--" if key == "cal" else "-"
            ax.plot(n_tests, means, color=colors[key], linestyle=ls,
                    marker="o", markersize=4, linewidth=1.5, label=labels[key])
            ax.fill_between(n_tests, means - stds, means + stds,
                            color=colors[key], alpha=0.1)

        # True MCE
        true_mces = [r["mean_true_mce"] for r in results_nc]
        ax.plot(n_tests, true_mces, color="black", linestyle=":",
                linewidth=1, label="True MCE")

        ax.set_xscale("log")
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(n_test_values)
        ax.set_xticklabels([str(v) for v in n_test_values], rotation=45, ha="right")
        ax.set_xlabel(r"$n_{\mathrm{test}}$")
        ax.set_ylabel("MCE certificate")
        ax.set_title(f"$n_{{\\mathrm{{cal}}}} = {n_cal}$", fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=7.5)

    fig.suptitle(f"MCE certificates: Cal-only vs Test-only vs Pooled "
                 f"(CP, α={alpha}, {n_repeats} repeats)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, f"{output_dir}/pooled_certificates_vs_ntest")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: Coverage vs n_test
    # =========================================================================
    fig, axes = plt.subplots(1, len(n_cal_values), figsize=(5 * len(n_cal_values), 4))

    if len(n_cal_values) == 1:
        axes = [axes]

    for ax_idx, n_cal in enumerate(n_cal_values):
        ax = axes[ax_idx]
        results_nc = [r for r in all_results if r["n_cal"] == n_cal]
        n_tests = [r["n_test"] for r in results_nc]

        for key in ["cal", "test", "pool", "best"]:
            covs = [r["coverage"][key] for r in results_nc]
            ls = "--" if key == "cal" else "-"
            ax.plot(n_tests, covs, color=colors[key], linestyle=ls,
                    marker="o", markersize=4, linewidth=1.5, label=labels[key])

        ax.axhline(1 - alpha, color="red", linestyle=":", linewidth=0.8,
                    label=f"Target ({1-alpha:.2f})")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(n_test_values)
        ax.set_xticklabels([str(v) for v in n_test_values], rotation=45, ha="right")
        ax.set_xlabel(r"$n_{\mathrm{test}}$")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.90, 1.005)
        ax.set_title(f"$n_{{\\mathrm{{cal}}}} = {n_cal}$", fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(f"Coverage of MCE certificates (target ≥ {1-alpha:.2f})",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, f"{output_dir}/pooled_coverage_vs_ntest")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: Ratio pooled/test-only vs n_test
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    cal_colors = {100: "#d62728", 500: "#2ca02c", 1000: "#1f77b4"}
    for n_cal in n_cal_values:
        results_nc = [r for r in all_results if r["n_cal"] == n_cal]
        n_tests = [r["n_test"] for r in results_nc]
        ratios = [r["mean_cert"]["pool"] / r["mean_cert"]["test"]
                  for r in results_nc]
        ax.plot(n_tests, ratios, color=cal_colors[n_cal],
                marker="o", markersize=5, linewidth=1.5,
                label=f"$n_{{\\mathrm{{cal}}}} = {n_cal}$")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(n_test_values)
    ax.set_xticklabels([str(v) for v in n_test_values], rotation=45, ha="right")
    ax.set_xlabel(r"$n_{\mathrm{test}}$")
    ax.set_ylabel("Ratio: Pooled / Test-only certificate")
    ax.set_title("Improvement of pooled over test-only", fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, f"{output_dir}/pooled_ratio_vs_ntest")
    plt.close(fig)

    # =========================================================================
    # Save results
    # =========================================================================
    json_path = os.path.join(output_dir, "pooled_bounds_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "alpha": alpha, "bound": bound, "n_repeats": n_repeats,
                "scenario": scenario, "n_cal_values": n_cal_values,
                "n_test_values": n_test_values,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved results to {json_path}")


if __name__ == "__main__":
    main()
