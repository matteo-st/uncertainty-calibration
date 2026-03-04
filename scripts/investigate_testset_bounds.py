#!/usr/bin/env python3
"""
Test-set concentration bounds for discrete calibrators (UM and isotonic step).

Since both UM and isotonic step produce discrete calibrated values, we can group
test samples by their discrete value and apply per-group concentration bounds.
These test-set bounds are much tighter than the n_cal-based theorem bounds because:
  1. n_test >> n_cal (7600 vs 1000)
  2. Fewer groups means more samples per group

Usage:
    python scripts/investigate_testset_bounds.py
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
# Concentration bound implementations (from investigate_concentration_bounds.py)
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


BOUND_NAMES = ["Hoeffding", "KL-Chernoff", "Clopper-Pearson"]
BOUND_COLORS = {
    "Hoeffding": "#d62728",
    "KL-Chernoff": "#2ca02c",
    "Clopper-Pearson": "#1f77b4",
}


def compute_bounds(p_hat, n_b, delta_b):
    """Compute epsilon from each bound for one group."""
    k = int(round(p_hat * n_b))
    k = max(0, min(k, n_b))
    return {
        "Hoeffding": hoeffding_epsilon(n_b, delta_b),
        "KL-Chernoff": kl_chernoff_epsilon(p_hat, n_b, delta_b),
        "Clopper-Pearson": clopper_pearson_epsilon(k, n_b, delta_b),
    }


# ---------------------------------------------------------------------------
# Analysis: per-group test-set bounds
# ---------------------------------------------------------------------------

RESULTS_DIR = "results/paper_metrics"
MODELS = ["electra", "bert", "deberta"]
DATASETS = ["mrpc", "sst2", "cola", "agnews"]
SEEDS = [42, 123, 456]
SCORES = ["sp", "md"]
DATASET_NAMES = {"mrpc": "MRPC", "sst2": "SST-2", "cola": "CoLA", "agnews": "AG News"}
MODEL_NAMES = {"electra": "ELECTRA", "bert": "BERT", "deberta": "DeBERTa"}

ALPHA = 0.05


def analyze_experiment(model, dataset, seed, score, alpha=ALPHA):
    """Load test scores and compute per-group test-set bounds."""
    npz_path = f"{RESULTS_DIR}/{model}_{dataset}_seed{seed}_test_scores.npz"
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    errors = data["errors"]
    calibrated = data[f"{score}_calibrated"]
    raw = data[f"{score}_raw"]
    n_test = len(errors)

    # Discrete groups from UM calibration
    unique_vals = np.sort(np.unique(calibrated))
    B = len(unique_vals)
    delta_b = alpha / B  # union bound

    groups = []
    for t in unique_vals:
        mask = calibrated == t
        n_b = mask.sum()
        k_b = int(errors[mask].sum())
        p_test = errors[mask].mean()  # test-set estimate of P(E=1|û=t)
        calib_error = abs(p_test - t)

        bounds = compute_bounds(p_test, n_b, delta_b)

        groups.append({
            "t": float(t),
            "n_b": int(n_b),
            "k_b": int(k_b),
            "p_test": float(p_test),
            "calib_error": float(calib_error),
            **{f"eps_{name}": float(bounds[name]) for name in BOUND_NAMES},
        })

    # MCE = max calibration error over groups (observed on test set)
    mce_observed = max(g["calib_error"] for g in groups)

    # CI width: max ε_b (how well we know p_b from test data)
    ci_width = {
        name: max(g[f"eps_{name}"] for g in groups) for name in BOUND_NAMES
    }

    # MCE certificate: max_b (|p_test,b - t_b| + ε_b)
    # This bounds the TRUE calibration error: |p_b - t_b| ≤ |p_test,b - t_b| + ε_b
    mce_cert = {
        name: max(g["calib_error"] + g[f"eps_{name}"] for g in groups)
        for name in BOUND_NAMES
    }

    # ECE certificate: sum_b (w_b * (|p_test,b - t_b| + ε_b))
    ece_observed = sum(g["n_b"] / n_test * g["calib_error"] for g in groups)
    ece_cert = {
        name: sum(g["n_b"] / n_test * (g["calib_error"] + g[f"eps_{name}"])
                  for g in groups)
        for name in BOUND_NAMES
    }

    # n_cal-based theorem bound for comparison
    n_cal = 1000
    B_um = max(2, int(2 * (n_cal ** (1/3))))
    n_b_cal = max(1, (n_cal + 1) // B_um - 1)
    delta_b_cal = alpha / B_um
    theorem_hoeff = hoeffding_epsilon(n_b_cal, delta_b_cal)

    return {
        "model": model, "dataset": dataset, "seed": seed, "score": score,
        "n_test": n_test, "B": B, "n_b_min": min(g["n_b"] for g in groups),
        "n_b_max": max(g["n_b"] for g in groups),
        "n_b_median": int(np.median([g["n_b"] for g in groups])),
        "mce_observed": mce_observed,
        "mce_cert_hoeff": mce_cert["Hoeffding"],
        "mce_cert_kl": mce_cert["KL-Chernoff"],
        "mce_cert_cp": mce_cert["Clopper-Pearson"],
        "ci_width_hoeff": ci_width["Hoeffding"],
        "ci_width_kl": ci_width["KL-Chernoff"],
        "ci_width_cp": ci_width["Clopper-Pearson"],
        "ece_observed": ece_observed,
        "ece_cert_hoeff": ece_cert["Hoeffding"],
        "ece_cert_kl": ece_cert["KL-Chernoff"],
        "ece_cert_cp": ece_cert["Clopper-Pearson"],
        "theorem_hoeff": theorem_hoeff,
        "groups": groups,
    }


def save_fig(fig, path_no_ext):
    """Save figure as both PDF and PNG."""
    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{path_no_ext}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {path_no_ext}.pdf/.png")


def main():
    output_dir = "results/testset_bounds"
    os.makedirs(output_dir, exist_ok=True)

    # Run all experiments
    all_results = []
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                for score in SCORES:
                    result = analyze_experiment(model, dataset, seed, score)
                    if result:
                        all_results.append(result)

    print(f"\nAnalyzed {len(all_results)} experiments\n")

    # =========================================================================
    # TABLE 1: MCE certificates — observed + CI = upper bound on true MCE
    # MCE_true ≤ max_b(|p_test,b - t_b| + ε_b)  with prob ≥ 1-α
    # =========================================================================
    print("=" * 110)
    print("TABLE 1: Test-set MCE certificates vs Theorem bound (n_cal=1000)")
    print("         MCE_cert = max_b(|p_test,b - t_b| + ε_b)  [upper bound on true MCE]")
    print("         (averaged over 3 models × 3 seeds = 9 experiments per cell)")
    print("=" * 110)
    print(f"{'Dataset':>8} {'Score':>5} {'B':>4} {'n_b med':>8} "
          f"{'MCE_obs':>8} "
          f"{'Cert_CP':>8} {'Cert_KL':>8} {'Cert_H':>8} "
          f"{'Thm(H)':>8}")
    print("-" * 110)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue
            B = np.mean([r["B"] for r in exps])
            nb_med = np.mean([r["n_b_median"] for r in exps])
            mce_obs = np.mean([r["mce_observed"] for r in exps])
            cert_h = np.mean([r["mce_cert_hoeff"] for r in exps])
            cert_kl = np.mean([r["mce_cert_kl"] for r in exps])
            cert_cp = np.mean([r["mce_cert_cp"] for r in exps])
            thm_h = np.mean([r["theorem_hoeff"] for r in exps])
            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} "
                  f"{B:>4.0f} {nb_med:>8.0f} "
                  f"{mce_obs:>8.4f} "
                  f"{cert_cp:>8.4f} {cert_kl:>8.4f} {cert_h:>8.4f} "
                  f"{thm_h:>8.4f}")

    # =========================================================================
    # TABLE 2: CI widths only (max ε_b) — how precisely we know p_b
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 2: Max CI width = max_b ε_b (estimation precision)")
    print("         (averaged over 9 experiments per cell)")
    print("=" * 100)
    print(f"{'Dataset':>8} {'Score':>5} {'n_b med':>8} "
          f"{'ε_Hoeff':>8} {'ε_KL':>8} {'ε_CP':>8} "
          f"{'Thm(H)':>8} {'CP/Thm':>8}")
    print("-" * 80)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue
            nb_med = np.mean([r["n_b_median"] for r in exps])
            w_h = np.mean([r["ci_width_hoeff"] for r in exps])
            w_kl = np.mean([r["ci_width_kl"] for r in exps])
            w_cp = np.mean([r["ci_width_cp"] for r in exps])
            thm_h = np.mean([r["theorem_hoeff"] for r in exps])
            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} "
                  f"{nb_med:>8.0f} "
                  f"{w_h:>8.4f} {w_kl:>8.4f} {w_cp:>8.4f} "
                  f"{thm_h:>8.4f} {w_cp/thm_h:>8.3f}")

    # =========================================================================
    # TABLE 3: ECE certificates
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 3: Test-set ECE certificates = Σ_b w_b(|p_test,b - t_b| + ε_b)")
    print("=" * 100)
    print(f"{'Dataset':>8} {'Score':>5} "
          f"{'ECE_obs':>8} "
          f"{'Cert_CP':>8} {'Cert_KL':>8} {'Cert_H':>8} "
          f"{'Thm(H)':>8}")
    print("-" * 80)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue
            ece_obs = np.mean([r["ece_observed"] for r in exps])
            cert_h = np.mean([r["ece_cert_hoeff"] for r in exps])
            cert_kl = np.mean([r["ece_cert_kl"] for r in exps])
            cert_cp = np.mean([r["ece_cert_cp"] for r in exps])
            thm_h = np.mean([r["theorem_hoeff"] for r in exps])
            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} "
                  f"{ece_obs:>8.4f} "
                  f"{cert_cp:>8.4f} {cert_kl:>8.4f} {cert_h:>8.4f} "
                  f"{thm_h:>8.4f}")

    # =========================================================================
    # TABLE 4: Per-group detail for DeBERTa SST-2 seed 42 (example)
    # =========================================================================
    example = next((r for r in all_results
                    if r["model"] == "deberta" and r["dataset"] == "sst2"
                    and r["seed"] == 42 and r["score"] == "sp"), None)
    if example:
        print("\n" + "=" * 110)
        print(f"TABLE 4: Per-group detail — DeBERTa, SST-2, seed 42, SP")
        print(f"         n_test={example['n_test']}, B={example['B']} groups, "
              f"α={ALPHA}")
        print(f"         Certificate for group b: |p_test,b - t_b| + ε_b")
        print("=" * 110)
        print(f"{'t_b':>8} {'n_b':>6} {'k_b':>5} {'p_test':>7} "
              f"{'|err|':>7} "
              f"{'ε_CP':>8} {'cert_CP':>8}")
        print("-" * 60)
        for g in example["groups"]:
            cert = g["calib_error"] + g["eps_Clopper-Pearson"]
            print(f"{g['t']:>8.4f} {g['n_b']:>6} {g['k_b']:>5} "
                  f"{g['p_test']:>7.4f} "
                  f"{g['calib_error']:>7.4f} "
                  f"{g['eps_Clopper-Pearson']:>8.4f} "
                  f"{cert:>8.4f}")
        print("-" * 60)
        print(f"{'MCE':>8} {'':>6} {'':>5} {'':>7} "
              f"{example['mce_observed']:>7.4f} "
              f"{'':>8} "
              f"{example['mce_cert_cp']:>8.4f}")
        print(f"\nTheorem bound (n_cal=1000, Hoeffding): {example['theorem_hoeff']:.4f}")
        print(f"Test-set certificate (CP):              {example['mce_cert_cp']:.4f}")
        print(f"Observed test MCE:                      {example['mce_observed']:.4f}")

    # =========================================================================
    # FIGURE 1: Per-group bounds vs actual calibration error (DeBERTa, all datasets)
    # =========================================================================
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for col_idx, dataset in enumerate(DATASETS):
        for row_idx, score in enumerate(SCORES):
            ax = axes[row_idx, col_idx]
            exps = [r for r in all_results
                    if r["model"] == "deberta" and r["dataset"] == dataset
                    and r["score"] == score]

            # Collect all groups across seeds
            all_groups = []
            for exp in exps:
                all_groups.extend(exp["groups"])

            if not all_groups:
                continue

            t_vals = [g["t"] for g in all_groups]
            calib_errors = [g["calib_error"] for g in all_groups]

            # Plot actual calibration errors
            ax.scatter(t_vals, calib_errors, c="black", s=15, zorder=5,
                       alpha=0.6, label=r"$|p_{\mathrm{test}} - t|$")

            # Plot bounds
            for bname in BOUND_NAMES:
                eps_vals = [g[f"eps_{bname}"] for g in all_groups]
                ax.scatter(t_vals, eps_vals, c=BOUND_COLORS[bname], s=10,
                           alpha=0.5, label=bname, marker="^", zorder=4)

            # Theorem bound (horizontal line)
            thm = exps[0]["theorem_hoeff"]
            ax.axhline(thm, color="gray", linestyle="--", linewidth=1,
                       label=f"Theorem (n_cal)", zorder=1)

            ax.set_ylim(-0.01, max(thm * 1.1, 0.3))
            ax.grid(True, alpha=0.3, linewidth=0.5)

            if row_idx == 0:
                ax.set_title(DATASET_NAMES[dataset], fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{score.upper()} — Bound / Error")
            if row_idx == 1:
                ax.set_xlabel(r"$t_b$ (calibrated value)")
            if row_idx == 0 and col_idx == 3:
                ax.legend(loc="upper left", fontsize=6.5, markerscale=1.2)

    fig.suptitle("Test-set bounds for UM calibration (DeBERTa, 3 seeds)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"{output_dir}/testset_bounds_pergroup")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: MCE comparison — test-set bounds vs theorem bound
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, score in enumerate(SCORES):
        ax = axes[ax_idx]
        datasets_plot = DATASETS
        x = np.arange(len(datasets_plot))
        width = 0.15

        # Gather data (averaged over models and seeds)
        mce_obs_vals = []
        test_hoeff_vals = []
        test_kl_vals = []
        test_cp_vals = []
        thm_vals = []

        for dataset in datasets_plot:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            mce_obs_vals.append(np.mean([r["mce_observed"] for r in exps]))
            test_hoeff_vals.append(np.mean([r["mce_cert_hoeff"] for r in exps]))
            test_kl_vals.append(np.mean([r["mce_cert_kl"] for r in exps]))
            test_cp_vals.append(np.mean([r["mce_cert_cp"] for r in exps]))
            thm_vals.append(np.mean([r["theorem_hoeff"] for r in exps]))

        ax.bar(x - 2*width, mce_obs_vals, width, label="Observed MCE",
               color="black", alpha=0.7)
        ax.bar(x - width, test_cp_vals, width, label="Test cert. (CP)",
               color=BOUND_COLORS["Clopper-Pearson"], alpha=0.7)
        ax.bar(x, test_kl_vals, width, label="Test cert. (KL)",
               color=BOUND_COLORS["KL-Chernoff"], alpha=0.7)
        ax.bar(x + width, test_hoeff_vals, width, label="Test cert. (Hoeff.)",
               color=BOUND_COLORS["Hoeffding"], alpha=0.7)
        ax.bar(x + 2*width, thm_vals, width, label="Theorem (n_cal)",
               color="gray", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_NAMES[d] for d in datasets_plot])
        ax.set_ylabel("MCE / Bound")
        ax.set_title(f"{score.upper()} score", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("MCE: Observed vs Test-set certificates vs Theorem bound",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, f"{output_dir}/testset_bounds_mce_comparison")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: Per-group bound tightness — n_b vs epsilon
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, score in enumerate(SCORES):
        ax = axes[ax_idx]

        for dataset in DATASETS:
            exps = [r for r in all_results
                    if r["model"] == "deberta" and r["dataset"] == dataset
                    and r["score"] == score]

            for exp in exps:
                for g in exp["groups"]:
                    # Plot CP bound vs n_b, colored by p_test
                    ax.scatter(g["n_b"], g["eps_Clopper-Pearson"],
                               c=g["p_test"], cmap="viridis", s=15,
                               alpha=0.6, vmin=0, vmax=0.4, zorder=3)
                    # Also show actual error as X
                    ax.scatter(g["n_b"], g["calib_error"],
                               c="red", s=8, marker="x", alpha=0.4, zorder=4)

        ax.set_xlabel(r"$n_b$ (group size)")
        ax.set_ylabel(r"$\epsilon$ / actual error")
        ax.set_title(f"{score.upper()} score", fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 0.4))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"$p_{\mathrm{test}}$ (test error rate)", fontsize=10)

    fig.suptitle("Test-set CP bound (circles) vs actual error (×) by group size",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    save_fig(fig, f"{output_dir}/testset_bounds_nb_vs_eps")
    plt.close(fig)

    # =========================================================================
    # Save all results to JSON
    # =========================================================================
    # Strip groups for the summary JSON
    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k != "groups"}
        summary.append(s)

    json_path = os.path.join(output_dir, "testset_bounds_analysis.json")
    with open(json_path, "w") as f:
        json.dump({"experiments": summary, "alpha": ALPHA}, f, indent=2)
    print(f"\nSaved {len(summary)} results to {json_path}")

    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Test-set CP certificate vs Theorem Hoeffding bound (n_cal=1000)")
    print("=" * 80)
    all_ratios = [r["mce_cert_cp"] / r["theorem_hoeff"] for r in all_results]
    print(f"  Avg ratio (test CP cert / theorem Hoeff): {np.mean(all_ratios):.3f}")
    print(f"  Range: [{min(all_ratios):.3f}, {max(all_ratios):.3f}]")
    # Exclude MRPC (tiny test set) for the main message
    large_test = [r for r in all_results if r["dataset"] in ("sst2", "agnews")]
    large_ratios = [r["mce_cert_cp"] / r["theorem_hoeff"] for r in large_test]
    print(f"\n  For large test sets (SST-2, AG News) only:")
    print(f"    Avg ratio: {np.mean(large_ratios):.3f}")
    print(f"    → Test-set CP certificate is {(1 - np.mean(large_ratios))*100:.0f}% "
          f"tighter than the theorem bound")


if __name__ == "__main__":
    main()
