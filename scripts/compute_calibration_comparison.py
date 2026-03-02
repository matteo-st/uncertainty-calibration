"""
Compare calibration methods: Uniform Mass vs Platt Scaling vs Isotonic Regression.

For each (model, dataset, seed) and each score, fits all three calibration methods
on the calibration set and evaluates on the test set.

Key difference from compute_paper_metrics.py: includes Platt and Isotonic baselines,
and uses a general Platt scaling (sigmoid(a*score + b)) that works for unbounded scores.

Usage:
    python scripts/compute_calibration_comparison.py \
        --cache_prefixes cache/paper/electra_mrpc_seed42 ... \
        --output_dir results/calibration_comparison
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    compute_rocauc,
    compute_ece_uniform_mass,
    compute_ece_discrete,
    compute_mce_uniform_mass,
    compute_mce_discrete,
)
from src.score_calibration import UniformMassCalibration
from src.uncertainty import (
    compute_max_proba_complement,
    compute_predictive_entropy,
    compute_doctor,
    compute_energy,
)

# Scores to compute (key, name, bounded, source)
SCORE_DEFS = [
    {"key": "sp", "name": "SP", "bounded": True, "source": "probs"},
    {"key": "pe", "name": "PE", "bounded": True, "source": "probs"},
    {"key": "doctor", "name": "Doctor", "bounded": True, "source": "probs"},
    {"key": "energy", "name": "Energy", "bounded": False, "source": "logits"},
    {"key": "md", "name": "MD", "bounded": False, "source": "cache"},
]


def compute_score(score_def, data):
    """Compute a score array from cached data."""
    key, source = score_def["key"], score_def["source"]
    if source == "probs":
        probs = data["probs"]
        if key == "sp":
            return compute_max_proba_complement(probs)
        elif key == "pe":
            return compute_predictive_entropy(probs)
        elif key == "doctor":
            return compute_doctor(probs)
    elif source == "logits":
        if key == "energy" and "logits" in data:
            return compute_energy(data["logits"])
    elif source == "cache":
        if key == "md":
            return data.get("md_scores")
    return None


def load_split(prefix, split):
    """Load npz file for a given prefix and split."""
    path = f"{prefix}_{split}.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}


# --- General Platt Scaling (works on any real-valued scores) ---
# Standardizes scores first to handle unbounded scores (Energy, MD).
# Fits: P(error | score) = sigmoid(a * z + b) where z = (score - mu) / sigma.

def fit_platt_general(cal_scores, cal_errors):
    """Fit Platt scaling with score standardization for numerical stability."""
    mu = cal_scores.mean()
    sigma = cal_scores.std()
    if sigma < 1e-12:
        sigma = 1.0
    z_cal = (cal_scores - mu) / sigma

    def neg_log_likelihood(params):
        a, b = params
        logits = a * z_cal + b
        probs = expit(logits)
        eps = 1e-10
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(
            cal_errors * np.log(probs) + (1 - cal_errors) * np.log(1 - probs)
        )

    mean_err = cal_errors.mean()
    b_init = np.log(mean_err / (1 - mean_err + 1e-10))
    result = minimize(neg_log_likelihood, [1.0, b_init], method="L-BFGS-B",
                      options={"maxiter": 5000})
    a, b = result.x
    return a, b, mu, sigma


def calibrate_platt_general(scores, a, b, mu, sigma):
    """Apply general Platt scaling with standardization."""
    z = (scores - mu) / sigma
    return expit(a * z + b)


# --- Temperature Scaling (a only, no bias) ---

def fit_temp_general(cal_scores, cal_errors):
    """Fit temperature scaling with score standardization."""
    mu = cal_scores.mean()
    sigma = cal_scores.std()
    if sigma < 1e-12:
        sigma = 1.0
    z_cal = (cal_scores - mu) / sigma

    def neg_log_likelihood(params):
        a = params[0]
        logits = a * z_cal
        probs = expit(logits)
        eps = 1e-10
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(
            cal_errors * np.log(probs) + (1 - cal_errors) * np.log(1 - probs)
        )

    result = minimize(neg_log_likelihood, [1.0], method="L-BFGS-B",
                      options={"maxiter": 5000})
    return result.x[0], mu, sigma


def calibrate_temp_general(scores, a, mu, sigma):
    """Apply temperature scaling with standardization."""
    z = (scores - mu) / sigma
    return expit(a * z)


# --- Metrics computation ---

def compute_metrics_for_method(test_cal_probs, test_errors, discrete=False):
    """Compute ROCAUC, ECE, MCE for calibrated test scores."""
    rocauc = compute_rocauc(test_cal_probs, test_errors)
    if discrete:
        ece = compute_ece_discrete(test_cal_probs, test_errors)
        mce = compute_mce_discrete(test_cal_probs, test_errors)
    else:
        ece = compute_ece_uniform_mass(test_cal_probs, test_errors)
        mce = compute_mce_uniform_mass(test_cal_probs, test_errors)
    return {"rocauc": rocauc, "ece": ece, "mce": mce}


def process_experiment(cache_prefix, score_defs):
    """Process one experiment: compare calibration methods for all scores."""
    cal_data = load_split(cache_prefix, "cal")
    test_data = load_split(cache_prefix, "test")

    cal_errors = (cal_data["predictions"] != cal_data["labels"]).astype(float)
    test_errors = (test_data["predictions"] != test_data["labels"]).astype(float)

    n_cal = len(cal_errors)
    n_test = len(test_errors)
    error_rate = float(test_errors.mean())

    results = {}

    for sdef in score_defs:
        key = sdef["key"]
        cal_scores = compute_score(sdef, cal_data)
        test_scores = compute_score(sdef, test_data)

        if cal_scores is None or test_scores is None:
            continue

        score_results = {}

        # --- Raw (before calibration) ---
        raw_metrics = {"rocauc": compute_rocauc(test_scores, test_errors)}
        if sdef["bounded"]:
            raw_metrics["ece"] = compute_ece_uniform_mass(test_scores, test_errors)
            raw_metrics["mce"] = compute_mce_uniform_mass(test_scores, test_errors)
        score_results["raw"] = raw_metrics

        # --- Uniform Mass ---
        um = UniformMassCalibration()
        um.fit(cal_scores, cal_errors)
        test_um = um.calibrate(test_scores)
        score_results["um"] = compute_metrics_for_method(test_um, test_errors, discrete=True)
        score_results["um"]["n_bins"] = len(um.bin_error_rates)

        # --- Platt Scaling (general) ---
        a, b, mu, sigma = fit_platt_general(cal_scores, cal_errors)
        test_platt = calibrate_platt_general(test_scores, a, b, mu, sigma)
        score_results["platt"] = compute_metrics_for_method(test_platt, test_errors, discrete=False)
        score_results["platt"]["params"] = {"a": float(a), "b": float(b),
                                            "mu": float(mu), "sigma": float(sigma)}

        # --- Temperature Scaling (general, no bias) ---
        a_temp, mu_t, sigma_t = fit_temp_general(cal_scores, cal_errors)
        test_temp = calibrate_temp_general(test_scores, a_temp, mu_t, sigma_t)
        score_results["temp"] = compute_metrics_for_method(test_temp, test_errors, discrete=False)
        score_results["temp"]["params"] = {"a": float(a_temp)}

        # --- Isotonic Regression ---
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_scores, cal_errors)
        test_iso = iso.predict(test_scores)
        score_results["isotonic"] = compute_metrics_for_method(test_iso, test_errors, discrete=False)

        results[key] = score_results

    meta = {
        "n_cal": n_cal,
        "n_test": n_test,
        "error_rate": error_rate,
    }
    return results, meta


def main():
    parser = argparse.ArgumentParser(description="Compare calibration methods")
    parser.add_argument("--cache_prefixes", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="results/calibration_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Theoretical epsilon for n_cal=1000, B=20, alpha=0.05
    # B = floor(2 * 1000^(1/3)) = 20
    # n_per_bin = floor((1000+1)/20) - 1 = 49
    # epsilon = sqrt(log(2*20/0.05) / (2*49))
    import math
    B_scott = int(2 * (1000 ** (1/3)))
    n_per_bin = (1000 + 1) // B_scott - 1
    epsilon_theory = math.sqrt(math.log(2 * B_scott / 0.05) / (2 * n_per_bin))
    print(f"Theoretical epsilon: {epsilon_theory:.4f} (B={B_scott}, n_per_bin={n_per_bin})")

    all_results = {}

    for prefix in args.cache_prefixes:
        exp_name = os.path.basename(prefix)
        print(f"Processing: {exp_name}")
        results, meta = process_experiment(prefix, SCORE_DEFS)
        meta["epsilon_theory"] = epsilon_theory
        all_results[exp_name] = {"scores": results, "meta": meta}

    # Save JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    json_path = os.path.join(args.output_dir, "calibration_comparison.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved: {json_path}")

    # Print summary table for 3 main scores
    print(f"\n{'='*90}")
    print("SUMMARY: SP score — ROCAUC / ECE / MCE by method")
    print(f"{'='*90}")
    print(f"{'Experiment':<30} {'Method':<10} {'ROCAUC':>8} {'ECE':>8} {'MCE':>8} {'MCE≤ε':>6}")
    print("-" * 78)
    for exp_name, data in sorted(all_results.items()):
        sp = data["scores"].get("sp", {})
        for method in ["raw", "um", "platt", "isotonic"]:
            m = sp.get(method, {})
            if not m:
                continue
            roc = m.get("rocauc", float("nan"))
            ece = m.get("ece", float("nan"))
            mce = m.get("mce", float("nan"))
            check = "✓" if mce <= epsilon_theory else "✗" if not np.isnan(mce) else "—"
            label = exp_name if method == "raw" else ""
            print(f"{label:<30} {method:<10} {roc:>8.4f} {ece:>8.4f} {mce:>8.4f} {check:>6}")
        print()


if __name__ == "__main__":
    main()
