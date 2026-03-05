#!/usr/bin/env python3
"""
Test-set concentration bounds: Isotonic Step vs Uniform Mass.

Compares test-set MCE/ECE certificates for two discrete calibrators:
  1. Uniform Mass (UM) — guarantees roughly equal mass per bin
  2. Isotonic Step — PAV breakpoints, NO mass control

Hypothesis: isotonic step produces groups with very unequal test mass,
so some groups have few samples → wide CIs → loose MCE certificate.

Runs from cache/paper/ (needs cal + test .npz files) — execute on server.

Usage:
    python scripts/investigate_testset_bounds_isotonic.py \
        --cache_dir cache/paper \
        --output_dir results/testset_bounds_isotonic
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Concentration bound implementations
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
# Calibrator implementations
# ---------------------------------------------------------------------------

def isotonic_predict_step(ir, X):
    """Predict using step function instead of linear interpolation."""
    if hasattr(ir, 'X_thresholds_') and hasattr(ir, 'y_thresholds_'):
        x_bp = ir.X_thresholds_
        y_bp = ir.y_thresholds_
    elif hasattr(ir, 'f_'):
        x_bp = ir.f_.x
        y_bp = ir.f_.y
    else:
        raise RuntimeError("Cannot extract breakpoints from IsotonicRegression")

    if len(x_bp) == 1:
        return np.full_like(X, y_bp[0], dtype=float)

    f_step = interp1d(
        x_bp, y_bp, kind='previous',
        bounds_error=False,
        fill_value=(y_bp[0], y_bp[-1])
    )
    result = f_step(X)
    result[X >= x_bp[-1]] = y_bp[-1]
    return result


def um_calibrate(cal_scores, cal_errors, test_scores):
    """Fit UM calibration (Scott's rule) and apply to test scores."""
    n_cal = len(cal_scores)
    B = max(2, int(2 * (n_cal ** (1/3))))

    # Quantile bin edges
    quantiles = np.linspace(0, 100, B + 1)
    bin_edges = np.percentile(cal_scores, quantiles)
    bin_edges = np.unique(bin_edges)
    actual_B = len(bin_edges) - 1

    # Compute per-bin error rates on cal data
    bin_error_rates = np.zeros(actual_B)
    for i in range(actual_B):
        if i == actual_B - 1:
            mask = (cal_scores >= bin_edges[i]) & (cal_scores <= bin_edges[i + 1])
        else:
            mask = (cal_scores >= bin_edges[i]) & (cal_scores < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_error_rates[i] = cal_errors[mask].mean()
        else:
            bin_error_rates[i] = cal_errors.mean()

    # Apply to test scores
    bin_indices = np.digitize(test_scores, bin_edges[1:-1])
    bin_indices = np.clip(bin_indices, 0, len(bin_error_rates) - 1)
    return bin_error_rates[bin_indices]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

MODELS = ["electra", "bert", "deberta"]
DATASETS = ["mrpc", "sst2", "cola", "agnews"]
SEEDS = [42, 123, 456]
SCORES = ["sp", "md"]
DATASET_NAMES = {"mrpc": "MRPC", "sst2": "SST-2", "cola": "CoLA", "agnews": "AG News"}
MODEL_NAMES = {"electra": "ELECTRA", "bert": "BERT", "deberta": "DeBERTa"}
ALPHA = 0.05


def extract_scores(data, score_name):
    """Extract uncertainty score from cached npz data."""
    if score_name == "sp":
        return 1.0 - np.max(data["probs"], axis=1)
    elif score_name == "md":
        return data["md_scores"].astype(float)
    raise ValueError(f"Unknown score: {score_name}")


def analyze_groups(calibrated_test, test_errors, alpha, method_name):
    """Group test samples by discrete calibrated value and compute bounds."""
    n_test = len(test_errors)
    unique_vals = np.sort(np.unique(calibrated_test))
    B = len(unique_vals)
    delta_b = alpha / B  # union bound

    groups = []
    for t in unique_vals:
        mask = calibrated_test == t
        n_b = int(mask.sum())
        k_b = int(test_errors[mask].sum())
        p_test = test_errors[mask].mean()
        calib_error = abs(p_test - t)

        bounds = compute_bounds(p_test, n_b, delta_b)

        groups.append({
            "t": float(t),
            "n_b": n_b,
            "k_b": k_b,
            "p_test": float(p_test),
            "calib_error": float(calib_error),
            **{f"eps_{name}": float(bounds[name]) for name in BOUND_NAMES},
        })

    # Observed metrics
    mce_observed = max(g["calib_error"] for g in groups)
    ece_observed = sum(g["n_b"] / n_test * g["calib_error"] for g in groups)

    # MCE certificate: max_b(|p_test,b - t_b| + ε_b)
    mce_cert = {
        name: max(g["calib_error"] + g[f"eps_{name}"] for g in groups)
        for name in BOUND_NAMES
    }

    # ECE certificate
    ece_cert = {
        name: sum(g["n_b"] / n_test * (g["calib_error"] + g[f"eps_{name}"])
                  for g in groups)
        for name in BOUND_NAMES
    }

    # CI width (max epsilon over groups)
    ci_width = {
        name: max(g[f"eps_{name}"] for g in groups) for name in BOUND_NAMES
    }

    return {
        "method": method_name,
        "n_test": n_test,
        "B": B,
        "n_b_min": min(g["n_b"] for g in groups),
        "n_b_max": max(g["n_b"] for g in groups),
        "n_b_median": int(np.median([g["n_b"] for g in groups])),
        "n_b_std": float(np.std([g["n_b"] for g in groups])),
        "mce_observed": float(mce_observed),
        "ece_observed": float(ece_observed),
        "mce_cert_hoeff": float(mce_cert["Hoeffding"]),
        "mce_cert_kl": float(mce_cert["KL-Chernoff"]),
        "mce_cert_cp": float(mce_cert["Clopper-Pearson"]),
        "ci_width_hoeff": float(ci_width["Hoeffding"]),
        "ci_width_kl": float(ci_width["KL-Chernoff"]),
        "ci_width_cp": float(ci_width["Clopper-Pearson"]),
        "ece_cert_hoeff": float(ece_cert["Hoeffding"]),
        "ece_cert_kl": float(ece_cert["KL-Chernoff"]),
        "ece_cert_cp": float(ece_cert["Clopper-Pearson"]),
        "groups": groups,
    }


def analyze_experiment(cache_dir, model, dataset, seed, score, alpha=ALPHA):
    """Fit both UM and isotonic step on cal, evaluate test-set bounds."""
    prefix = f"{cache_dir}/{model}_{dataset}_seed{seed}"
    cal_path = f"{prefix}_cal.npz"
    test_path = f"{prefix}_test.npz"

    for p in [cal_path, test_path]:
        if not os.path.exists(p):
            print(f"  SKIP: {p} not found")
            return None

    cal_data = {k: np.load(cal_path)[k] for k in np.load(cal_path).files}
    test_data = {k: np.load(test_path)[k] for k in np.load(test_path).files}

    cal_scores = extract_scores(cal_data, score)
    test_scores = extract_scores(test_data, score)
    cal_errors = (cal_data["predictions"] != cal_data["labels"]).astype(float)
    test_errors = (test_data["predictions"] != test_data["labels"]).astype(float)
    n_cal = len(cal_errors)

    # --- Isotonic step ---
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(cal_scores, cal_errors)
    cal_step = isotonic_predict_step(ir, test_scores)
    iso_result = analyze_groups(cal_step, test_errors, alpha, "isotonic_step")

    # --- Uniform Mass ---
    cal_um = um_calibrate(cal_scores, cal_errors, test_scores)
    um_result = analyze_groups(cal_um, test_errors, alpha, "um")

    # Theorem bound for reference
    B_um = max(2, int(2 * (n_cal ** (1/3))))
    n_b_cal = max(1, (n_cal + 1) // B_um - 1)
    delta_b_cal = alpha / B_um
    theorem_hoeff = hoeffding_epsilon(n_b_cal, delta_b_cal)

    return {
        "model": model, "dataset": dataset, "seed": seed, "score": score,
        "n_cal": n_cal, "n_test": iso_result["n_test"],
        "theorem_hoeff": float(theorem_hoeff),
        "isotonic": iso_result,
        "um": um_result,
    }


def save_fig(fig, path_no_ext):
    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{path_no_ext}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {path_no_ext}.pdf/.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="cache/paper")
    parser.add_argument("--output_dir", type=str, default="results/testset_bounds_isotonic")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Run all experiments
    all_results = []
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                for score in SCORES:
                    print(f"  {model} {dataset} seed={seed} {score}...", end=" ")
                    result = analyze_experiment(
                        args.cache_dir, model, dataset, seed, score
                    )
                    if result:
                        all_results.append(result)
                        iso = result["isotonic"]
                        um = result["um"]
                        print(f"ISO: B={iso['B']}, n_b=[{iso['n_b_min']},"
                              f"{iso['n_b_max']}] | "
                              f"UM: B={um['B']}, n_b=[{um['n_b_min']},"
                              f"{um['n_b_max']}]")
                    else:
                        print("SKIPPED")

    print(f"\nAnalyzed {len(all_results)} experiments\n")

    if not all_results:
        print("No results. Check --cache_dir path.")
        return

    # =========================================================================
    # TABLE 1: Group size statistics — Isotonic vs UM
    # =========================================================================
    print("=" * 120)
    print("TABLE 1: Group size distribution — Isotonic Step vs Uniform Mass")
    print("         (averaged over 3 models × 3 seeds = 9 experiments per cell)")
    print("=" * 120)
    print(f"{'Dataset':>8} {'Score':>5} {'n_test':>6} "
          f"│ {'B_iso':>5} {'min':>5} {'med':>5} {'max':>6} {'std':>6} "
          f"│ {'B_um':>5} {'min':>5} {'med':>5} {'max':>6} {'std':>6}")
    print("-" * 120)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue
            n_test = exps[0]["n_test"]
            # Isotonic
            iso_B = np.mean([r["isotonic"]["B"] for r in exps])
            iso_min = np.mean([r["isotonic"]["n_b_min"] for r in exps])
            iso_med = np.mean([r["isotonic"]["n_b_median"] for r in exps])
            iso_max = np.mean([r["isotonic"]["n_b_max"] for r in exps])
            iso_std = np.mean([r["isotonic"]["n_b_std"] for r in exps])
            # UM
            um_B = np.mean([r["um"]["B"] for r in exps])
            um_min = np.mean([r["um"]["n_b_min"] for r in exps])
            um_med = np.mean([r["um"]["n_b_median"] for r in exps])
            um_max = np.mean([r["um"]["n_b_max"] for r in exps])
            um_std = np.mean([r["um"]["n_b_std"] for r in exps])
            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} {n_test:>6} "
                  f"│ {iso_B:>5.0f} {iso_min:>5.0f} {iso_med:>5.0f} "
                  f"{iso_max:>6.0f} {iso_std:>6.0f} "
                  f"│ {um_B:>5.0f} {um_min:>5.0f} {um_med:>5.0f} "
                  f"{um_max:>6.0f} {um_std:>6.0f}")

    # =========================================================================
    # TABLE 2: MCE certificates — Isotonic vs UM vs Theorem
    # =========================================================================
    print("\n" + "=" * 130)
    print("TABLE 2: MCE certificates — Isotonic Step vs Uniform Mass vs Theorem")
    print("         MCE_cert = max_b(|p_test,b - t_b| + ε_b)")
    print("         (averaged over 9 experiments per cell)")
    print("=" * 130)
    print(f"{'Dataset':>8} {'Score':>5} "
          f"│ {'MCE_iso':>8} {'Cert_iso':>9} "
          f"│ {'MCE_um':>8} {'Cert_um':>9} "
          f"│ {'Thm(H)':>8} "
          f"│ {'iso/thm':>8} {'um/thm':>8}")
    print("-" * 130)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue
            mce_iso = np.mean([r["isotonic"]["mce_observed"] for r in exps])
            cert_iso = np.mean([r["isotonic"]["mce_cert_cp"] for r in exps])
            mce_um = np.mean([r["um"]["mce_observed"] for r in exps])
            cert_um = np.mean([r["um"]["mce_cert_cp"] for r in exps])
            thm = np.mean([r["theorem_hoeff"] for r in exps])
            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} "
                  f"│ {mce_iso:>8.4f} {cert_iso:>9.4f} "
                  f"│ {mce_um:>8.4f} {cert_um:>9.4f} "
                  f"│ {thm:>8.4f} "
                  f"│ {cert_iso/thm:>8.3f} {cert_um/thm:>8.3f}")

    # =========================================================================
    # TABLE 3: CI widths — max ε_b for isotonic vs UM
    # =========================================================================
    print("\n" + "=" * 110)
    print("TABLE 3: Max CI width = max_b ε_b (Clopper-Pearson)")
    print("         Driven by the SMALLEST group")
    print("         (averaged over 9 experiments per cell)")
    print("=" * 110)
    print(f"{'Dataset':>8} {'Score':>5} "
          f"│ {'n_b_min_iso':>11} {'ε_iso':>8} "
          f"│ {'n_b_min_um':>11} {'ε_um':>8} "
          f"│ {'Thm(H)':>8} "
          f"│ {'ε_iso/ε_um':>10}")
    print("-" * 110)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue
            nb_min_iso = np.mean([r["isotonic"]["n_b_min"] for r in exps])
            ci_iso = np.mean([r["isotonic"]["ci_width_cp"] for r in exps])
            nb_min_um = np.mean([r["um"]["n_b_min"] for r in exps])
            ci_um = np.mean([r["um"]["ci_width_cp"] for r in exps])
            thm = np.mean([r["theorem_hoeff"] for r in exps])
            ratio = ci_iso / ci_um if ci_um > 0 else float('inf')
            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} "
                  f"│ {nb_min_iso:>11.0f} {ci_iso:>8.4f} "
                  f"│ {nb_min_um:>11.0f} {ci_um:>8.4f} "
                  f"│ {thm:>8.4f} "
                  f"│ {ratio:>10.2f}")

    # =========================================================================
    # TABLE 4: ECE certificates
    # =========================================================================
    print("\n" + "=" * 110)
    print("TABLE 4: ECE certificates (Clopper-Pearson)")
    print("         ECE_cert = Σ_b w_b(|p_test,b - t_b| + ε_b)")
    print("=" * 110)
    print(f"{'Dataset':>8} {'Score':>5} "
          f"│ {'ECE_iso':>8} {'Cert_iso':>9} "
          f"│ {'ECE_um':>8} {'Cert_um':>9} "
          f"│ {'Thm(H)':>8}")
    print("-" * 110)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue
            ece_iso = np.mean([r["isotonic"]["ece_observed"] for r in exps])
            cert_iso = np.mean([r["isotonic"]["ece_cert_cp"] for r in exps])
            ece_um = np.mean([r["um"]["ece_observed"] for r in exps])
            cert_um = np.mean([r["um"]["ece_cert_cp"] for r in exps])
            thm = np.mean([r["theorem_hoeff"] for r in exps])
            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} "
                  f"│ {ece_iso:>8.4f} {cert_iso:>9.4f} "
                  f"│ {ece_um:>8.4f} {cert_um:>9.4f} "
                  f"│ {thm:>8.4f}")

    # =========================================================================
    # TABLE 5: Per-group detail — DeBERTa SST-2 seed 42 SP (example)
    # =========================================================================
    example = next((r for r in all_results
                    if r["model"] == "deberta" and r["dataset"] == "sst2"
                    and r["seed"] == 42 and r["score"] == "sp"), None)
    if example:
        for method_key, method_label in [("isotonic", "Isotonic Step"),
                                          ("um", "Uniform Mass")]:
            res = example[method_key]
            print(f"\n{'=' * 100}")
            print(f"TABLE 5{('a' if method_key == 'isotonic' else 'b')}: "
                  f"Per-group detail — {method_label} — "
                  f"DeBERTa, SST-2, seed 42, SP")
            print(f"  n_test={res['n_test']}, B={res['B']} groups, α={ALPHA}")
            print(f"{'=' * 100}")
            print(f"{'t_b':>8} {'n_b':>6} {'%test':>6} {'k_b':>5} "
                  f"{'p_test':>7} {'|err|':>7} "
                  f"{'ε_CP':>8} {'cert_CP':>8}")
            print("-" * 70)
            for g in res["groups"]:
                cert = g["calib_error"] + g["eps_Clopper-Pearson"]
                pct = 100 * g["n_b"] / res["n_test"]
                print(f"{g['t']:>8.4f} {g['n_b']:>6} {pct:>5.1f}% {g['k_b']:>5} "
                      f"{g['p_test']:>7.4f} {g['calib_error']:>7.4f} "
                      f"{g['eps_Clopper-Pearson']:>8.4f} {cert:>8.4f}")
            print("-" * 70)
            print(f"{'MCE':>8} {'':>6} {'':>6} {'':>5} "
                  f"{'':>7} {res['mce_observed']:>7.4f} "
                  f"{'':>8} {res['mce_cert_cp']:>8.4f}")

        print(f"\nTheorem bound (n_cal={example['n_cal']}, Hoeffding): "
              f"{example['theorem_hoeff']:.4f}")

    # =========================================================================
    # TABLE 6: Worst group analysis — what drives the isotonic MCE certificate?
    # =========================================================================
    print("\n" + "=" * 120)
    print("TABLE 6: Worst group analysis — group driving MCE certificate (CP)")
    print("         For each experiment, which group has the largest cert_b?")
    print("         (averaged over 9 experiments per cell)")
    print("=" * 120)
    print(f"{'Dataset':>8} {'Score':>5} "
          f"│ {'Iso worst n_b':>14} {'Iso worst t':>12} {'Iso cert':>9} "
          f"│ {'UM worst n_b':>13} {'UM worst t':>11} {'UM cert':>8}")
    print("-" * 120)

    for dataset in DATASETS:
        for score in SCORES:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            if not exps:
                continue

            iso_worst_nbs, iso_worst_ts, iso_certs = [], [], []
            um_worst_nbs, um_worst_ts, um_certs = [], [], []

            for r in exps:
                # Find worst group for isotonic
                iso_groups = r["isotonic"]["groups"]
                iso_certs_per_g = [g["calib_error"] + g["eps_Clopper-Pearson"]
                                   for g in iso_groups]
                worst_idx = np.argmax(iso_certs_per_g)
                iso_worst_nbs.append(iso_groups[worst_idx]["n_b"])
                iso_worst_ts.append(iso_groups[worst_idx]["t"])
                iso_certs.append(iso_certs_per_g[worst_idx])

                # Find worst group for UM
                um_groups = r["um"]["groups"]
                um_certs_per_g = [g["calib_error"] + g["eps_Clopper-Pearson"]
                                  for g in um_groups]
                worst_idx = np.argmax(um_certs_per_g)
                um_worst_nbs.append(um_groups[worst_idx]["n_b"])
                um_worst_ts.append(um_groups[worst_idx]["t"])
                um_certs.append(um_certs_per_g[worst_idx])

            print(f"{DATASET_NAMES[dataset]:>8} {score.upper():>5} "
                  f"│ {np.mean(iso_worst_nbs):>14.0f} "
                  f"{np.mean(iso_worst_ts):>12.4f} "
                  f"{np.mean(iso_certs):>9.4f} "
                  f"│ {np.mean(um_worst_nbs):>13.0f} "
                  f"{np.mean(um_worst_ts):>11.4f} "
                  f"{np.mean(um_certs):>8.4f}")

    # =========================================================================
    # FIGURE 1: Group size distribution — Isotonic vs UM (DeBERTa, all datasets)
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

            # Collect all group sizes across seeds for DeBERTa
            iso_nbs, um_nbs = [], []
            for r in all_results:
                if (r["model"] == "deberta" and r["dataset"] == dataset
                        and r["score"] == score):
                    iso_nbs.extend([g["n_b"] for g in r["isotonic"]["groups"]])
                    um_nbs.extend([g["n_b"] for g in r["um"]["groups"]])

            if not iso_nbs:
                continue

            n_test = next(r["n_test"] for r in all_results
                          if r["dataset"] == dataset)

            # Side-by-side bar chart of sorted group sizes
            iso_nbs_sorted = sorted(iso_nbs, reverse=True)
            um_nbs_sorted = sorted(um_nbs, reverse=True)

            max_groups = max(len(iso_nbs_sorted), len(um_nbs_sorted))
            x = np.arange(max_groups)
            width = 0.35

            ax.bar(x[:len(iso_nbs_sorted)] - width/2, iso_nbs_sorted,
                   width, label="Isotonic", color="#d62728", alpha=0.7)
            ax.bar(x[:len(um_nbs_sorted)] + width/2, um_nbs_sorted,
                   width, label="UM", color="#1f77b4", alpha=0.7)

            # Reference line: uniform distribution
            if um_nbs:
                avg_um = n_test / (len(um_nbs) // 3)  # 3 seeds
                ax.axhline(avg_um, color="gray", linestyle="--", linewidth=0.8,
                           label=f"n_test/B={avg_um:.0f}")

            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, axis="y")
            if row_idx == 0:
                ax.set_title(DATASET_NAMES[dataset], fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{score.upper()} — Group size $n_b$")
            if row_idx == 1:
                ax.set_xlabel("Group index (sorted by size)")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Group size distribution: Isotonic Step vs Uniform Mass (DeBERTa, 3 seeds)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"{output_dir}/group_sizes_comparison")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: MCE certificate comparison — bar chart
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, score in enumerate(SCORES):
        ax = axes[ax_idx]
        x = np.arange(len(DATASETS))
        width = 0.15

        mce_iso_obs, mce_um_obs = [], []
        cert_iso, cert_um, thm_vals = [], [], []

        for dataset in DATASETS:
            exps = [r for r in all_results
                    if r["dataset"] == dataset and r["score"] == score]
            mce_iso_obs.append(np.mean([r["isotonic"]["mce_observed"] for r in exps]))
            mce_um_obs.append(np.mean([r["um"]["mce_observed"] for r in exps]))
            cert_iso.append(np.mean([r["isotonic"]["mce_cert_cp"] for r in exps]))
            cert_um.append(np.mean([r["um"]["mce_cert_cp"] for r in exps]))
            thm_vals.append(np.mean([r["theorem_hoeff"] for r in exps]))

        ax.bar(x - 2*width, mce_iso_obs, width, label="MCE obs. (iso)",
               color="#d62728", alpha=0.5)
        ax.bar(x - width, cert_iso, width, label="Cert. CP (iso)",
               color="#d62728", alpha=0.85)
        ax.bar(x, mce_um_obs, width, label="MCE obs. (UM)",
               color="#1f77b4", alpha=0.5)
        ax.bar(x + width, cert_um, width, label="Cert. CP (UM)",
               color="#1f77b4", alpha=0.85)
        ax.bar(x + 2*width, thm_vals, width, label="Theorem (Hoeff)",
               color="gray", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_NAMES[d] for d in DATASETS])
        ax.set_ylabel("MCE / Certificate")
        ax.set_title(f"{score.upper()} score", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if ax_idx == 0:
            ax.legend(fontsize=6.5, loc="upper left")

    fig.suptitle("MCE certificates: Isotonic Step vs UM vs Theorem",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, f"{output_dir}/mce_comparison_iso_vs_um")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: Per-group bounds — isotonic vs UM side by side (DeBERTa SST-2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col_idx, (method_key, method_label) in enumerate(
            [("isotonic", "Isotonic Step"), ("um", "Uniform Mass")]):
        for row_idx, score in enumerate(SCORES):
            ax = axes[row_idx, col_idx]

            for r in all_results:
                if not (r["model"] == "deberta" and r["dataset"] == "sst2"
                        and r["score"] == score):
                    continue
                groups = r[method_key]["groups"]
                t_vals = [g["t"] for g in groups]
                calib_errors = [g["calib_error"] for g in groups]
                n_bs = [g["n_b"] for g in groups]

                # Size of scatter point proportional to n_b
                sizes = [max(5, n / 20) for n in n_bs]

                ax.scatter(t_vals, calib_errors, c="black", s=sizes,
                           alpha=0.5, zorder=5,
                           label=r"$|p_{\mathrm{test}} - t|$"
                           if r["seed"] == 42 else None)

                for bname in ["Clopper-Pearson"]:
                    eps_vals = [g[f"eps_{bname}"] for g in groups]
                    ax.scatter(t_vals, eps_vals, c=BOUND_COLORS[bname],
                               s=sizes, alpha=0.5, marker="^", zorder=4,
                               label=f"ε (CP)" if r["seed"] == 42 else None)

            thm = next(r["theorem_hoeff"] for r in all_results
                       if r["dataset"] == "sst2")
            ax.axhline(thm, color="gray", linestyle="--", linewidth=1,
                       label="Theorem (n_cal)", zorder=1)

            ax.set_ylim(-0.01, max(thm * 1.1, 0.35))
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if row_idx == 0:
                ax.set_title(f"{method_label} — SST-2", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{score.upper()} — Bound / Error")
            if row_idx == 1:
                ax.set_xlabel(r"$t_b$ (calibrated value)")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, markerscale=1.2)

    fig.suptitle("Per-group test-set bounds: Isotonic Step vs UM "
                 "(DeBERTa, SST-2, 3 seeds)\n"
                 "Point size ∝ group size $n_b$",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, f"{output_dir}/pergroup_iso_vs_um_sst2")
    plt.close(fig)

    # =========================================================================
    # FIGURE 4: n_b vs ε (CP) — isotonic vs UM overlay
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, score in enumerate(SCORES):
        ax = axes[ax_idx]

        for r in all_results:
            if r["model"] != "deberta" or r["score"] != score:
                continue

            for method_key, color, marker, label_prefix in [
                    ("isotonic", "#d62728", "o", "Iso"),
                    ("um", "#1f77b4", "s", "UM")]:
                groups = r[method_key]["groups"]
                nbs = [g["n_b"] for g in groups]
                eps_cps = [g["eps_Clopper-Pearson"] for g in groups]

                ds_label = DATASET_NAMES[r["dataset"]]
                label = (f"{label_prefix} ({ds_label})"
                         if r["seed"] == 42 else None)
                ax.scatter(nbs, eps_cps, c=color, marker=marker, s=15,
                           alpha=0.4, label=label, zorder=3)

        ax.set_xlabel(r"$n_b$ (group size)")
        ax.set_ylabel(r"$\epsilon_b$ (CP)")
        ax.set_title(f"{score.upper()} score", fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if ax_idx == 0:
            ax.legend(fontsize=6.5, ncol=2, loc="upper right")

    fig.suptitle("CP bound vs group size: Isotonic Step (red) vs UM (blue)\n"
                 "DeBERTa, all datasets, 3 seeds",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, f"{output_dir}/nb_vs_eps_iso_vs_um")
    plt.close(fig)

    # =========================================================================
    # FIGURE 5: Group size distribution as box plots
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, score in enumerate(SCORES):
        ax = axes[ax_idx]

        box_data = []
        box_labels = []

        for dataset in DATASETS:
            iso_nbs = []
            um_nbs = []
            for r in all_results:
                if (r["dataset"] == dataset and r["score"] == score):
                    iso_nbs.extend([g["n_b"] for g in r["isotonic"]["groups"]])
                    um_nbs.extend([g["n_b"] for g in r["um"]["groups"]])

            if iso_nbs:
                box_data.append(iso_nbs)
                box_labels.append(f"{DATASET_NAMES[dataset]}\nIso")
                box_data.append(um_nbs)
                box_labels.append(f"{DATASET_NAMES[dataset]}\nUM")

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                            showfliers=True, flierprops={"markersize": 3})
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor("#d62728" if i % 2 == 0 else "#1f77b4")
                patch.set_alpha(0.5)

        ax.set_ylabel(r"Group size $n_b$")
        ax.set_title(f"{score.upper()} score", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_yscale("log")

    fig.suptitle("Group size distribution: Isotonic Step (red) vs UM (blue)\n"
                 "All models, 3 seeds",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, f"{output_dir}/group_size_boxplots")
    plt.close(fig)

    # =========================================================================
    # Save all results to JSON
    # =========================================================================
    def strip_groups(result):
        """Create summary without per-group detail."""
        out = {}
        for k, v in result.items():
            if k in ("isotonic", "um"):
                out[k] = {kk: vv for kk, vv in v.items() if kk != "groups"}
            else:
                out[k] = v
        return out

    # Also save a version WITH groups for detailed analysis
    json_full_path = os.path.join(output_dir, "testset_bounds_isotonic_full.json")
    with open(json_full_path, "w") as f:
        # Convert groups for JSON serialization
        serializable = []
        for r in all_results:
            s = {k: v for k, v in r.items()}
            serializable.append(s)
        json.dump({"experiments": serializable, "alpha": ALPHA}, f, indent=2)
    print(f"\nSaved {len(all_results)} full results to {json_full_path}")

    json_path = os.path.join(output_dir, "testset_bounds_isotonic_summary.json")
    with open(json_path, "w") as f:
        summary = [strip_groups(r) for r in all_results]
        json.dump({"experiments": summary, "alpha": ALPHA}, f, indent=2)
    print(f"Saved {len(summary)} summary results to {json_path}")

    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Isotonic Step vs UM test-set certificates (CP, α=0.05)")
    print("=" * 80)

    for ds_group, ds_list in [("Large test (SST-2, AG News)",
                                ["sst2", "agnews"]),
                               ("Small test (MRPC, CoLA)",
                                ["mrpc", "cola"]),
                               ("All datasets", DATASETS)]:
        iso_certs = [r["isotonic"]["mce_cert_cp"] for r in all_results
                     if r["dataset"] in ds_list]
        um_certs = [r["um"]["mce_cert_cp"] for r in all_results
                    if r["dataset"] in ds_list]
        thm_vals = [r["theorem_hoeff"] for r in all_results
                    if r["dataset"] in ds_list]

        if not iso_certs:
            continue

        print(f"\n  {ds_group}:")
        print(f"    Isotonic MCE cert (CP): {np.mean(iso_certs):.4f} "
              f"± {np.std(iso_certs):.4f}")
        print(f"    UM MCE cert (CP):       {np.mean(um_certs):.4f} "
              f"± {np.std(um_certs):.4f}")
        print(f"    Theorem (Hoeff):        {np.mean(thm_vals):.4f}")
        print(f"    Iso cert / UM cert:     {np.mean(iso_certs)/np.mean(um_certs):.3f}")

    # Min group sizes
    print("\n  Minimum group sizes (averaged over experiments):")
    for dataset in DATASETS:
        exps = [r for r in all_results if r["dataset"] == dataset]
        if not exps:
            continue
        iso_min = np.mean([r["isotonic"]["n_b_min"] for r in exps])
        um_min = np.mean([r["um"]["n_b_min"] for r in exps])
        print(f"    {DATASET_NAMES[dataset]:>8}: Iso min={iso_min:>6.0f}, "
              f"UM min={um_min:>6.0f}, ratio={iso_min/um_min:.2f}")


if __name__ == "__main__":
    main()
