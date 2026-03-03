#!/usr/bin/env python3
"""
n_cal ablation: evaluate calibration methods across calibration set sizes.

For each n_cal value, samples calibration sets from the pool (original cal +
unused training data), fits UM/Platt/Isotonic on that sample, and evaluates
on the fixed test set. Runs both MD and SP scores.

Requires: cache_unused_samples.py must have been run first to produce
the *_unused.npz files for SST-2 and AG News.

Usage:
    python scripts/run_ncal_ablation.py \
        --cache_dir cache/paper \
        --output_dir results/ncal_ablation \
        --n_draws 20
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from src.evaluation import (
    compute_rocauc,
    compute_mce_discrete,
    compute_mce_uniform_mass,
)
from src.score_calibration import UniformMassCalibration

# --- Configuration ---

DATASETS = ["sst2", "agnews"]
MODEL_SHORT_NAMES = ["electra", "bert", "deberta"]
SEEDS = [42, 123, 456]
N_CAL_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
SCORES = ["md", "sp"]
ALPHA = 0.05  # confidence level for UM guarantee


# --- Platt scaling (reused from compute_calibration_comparison.py) ---

def fit_platt_general(cal_scores, cal_errors):
    """Fit Platt scaling with score standardization."""
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
    b_init = np.log(max(mean_err, 1e-6) / max(1 - mean_err, 1e-6))
    result = minimize(neg_log_likelihood, [1.0, b_init], method="L-BFGS-B",
                      options={"maxiter": 5000})
    a, b = result.x
    return a, b, mu, sigma


def calibrate_platt_general(scores, a, b, mu, sigma):
    """Apply general Platt scaling with standardization."""
    z = (scores - mu) / sigma
    return expit(a * z + b)


# --- Theoretical epsilon ---

def compute_theoretical_epsilon(n_cal, alpha=0.05):
    """Compute theoretical UM guarantee epsilon for given n_cal."""
    B = int(2 * (n_cal ** (1 / 3)))
    if B < 1:
        return float("inf")
    n_per_bin = (n_cal + 1) // B - 1
    if n_per_bin < 1:
        return float("inf")
    epsilon = math.sqrt(math.log(2 * B / alpha) / (2 * n_per_bin))
    return epsilon


# --- Score extraction ---

def extract_sp_scores(data):
    """Extract softmax probability score: SP = 1 - max(probs)."""
    return 1.0 - np.max(data["probs"], axis=1)


def extract_scores(data, score_name):
    """Extract scores from cached data by name."""
    if score_name == "md":
        return data["md_scores"]
    elif score_name == "sp":
        return extract_sp_scores(data)
    else:
        raise ValueError(f"Unknown score: {score_name}")


# --- Main ablation logic ---

def load_npz(path):
    """Load .npz and return as dict."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def run_ablation(cache_dir, n_cal_values, n_draws, output_dir):
    """Run the full n_cal ablation experiment."""
    results = []

    # Precompute theoretical epsilon for each n_cal
    theoretical_epsilon = {}
    for n_cal in n_cal_values:
        theoretical_epsilon[n_cal] = compute_theoretical_epsilon(n_cal, ALPHA)
        print(f"  n_cal={n_cal}: B={int(2*(n_cal**(1/3)))}, "
              f"epsilon={theoretical_epsilon[n_cal]:.4f}")

    total_combos = len(DATASETS) * len(MODEL_SHORT_NAMES) * len(SEEDS)
    combo_idx = 0

    for dataset in DATASETS:
        for model_short in MODEL_SHORT_NAMES:
            for seed in SEEDS:
                combo_idx += 1
                variant = f"{model_short}_{dataset}"
                prefix = str(cache_dir / f"{variant}_seed{seed}")

                print(f"\n[{combo_idx}/{total_combos}] {variant} seed={seed}")

                # Load cached data
                cal_path = f"{prefix}_cal.npz"
                unused_path = f"{prefix}_unused.npz"
                test_path = f"{prefix}_test.npz"

                if not Path(unused_path).exists():
                    print(f"  SKIP: {unused_path} not found")
                    continue

                cal_data = load_npz(cal_path)
                unused_data = load_npz(unused_path)
                test_data = load_npz(test_path)

                # Errors for pool and test (shared across scores)
                pool_errors = np.concatenate([
                    (cal_data["predictions"] != cal_data["labels"]).astype(float),
                    (unused_data["predictions"] != unused_data["labels"]).astype(float),
                ])
                test_errors = (
                    test_data["predictions"] != test_data["labels"]
                ).astype(float)

                for score_name in SCORES:
                    # Build score pools and test scores
                    pool_scores = np.concatenate([
                        extract_scores(cal_data, score_name),
                        extract_scores(unused_data, score_name),
                    ])
                    test_scores = extract_scores(test_data, score_name)
                    pool_size = len(pool_scores)

                    # Raw ROCAUC (before any calibration)
                    raw_rocauc = compute_rocauc(test_scores, test_errors)

                    print(f"  [{score_name.upper()}] Pool: {pool_size}, "
                          f"test: {len(test_scores)}, raw ROCAUC: {raw_rocauc:.4f}")

                    for n_cal in n_cal_values:
                        if n_cal > pool_size:
                            print(f"    SKIP n_cal={n_cal} > pool_size={pool_size}")
                            continue

                        for draw in range(n_draws):
                            # Sample n_cal from pool (same RNG per draw for reproducibility)
                            rng = np.random.RandomState(seed * 1000 + draw)
                            idx = rng.choice(pool_size, n_cal, replace=False)
                            cal_scores_sample = pool_scores[idx]
                            cal_errors_sample = pool_errors[idx]

                            # Skip if all errors are same class
                            if len(np.unique(cal_errors_sample)) < 2:
                                continue

                            base_entry = {
                                "dataset": dataset,
                                "model": model_short,
                                "seed": seed,
                                "score": score_name,
                                "n_cal": n_cal,
                                "draw": draw,
                                "raw_rocauc": float(raw_rocauc),
                            }

                            # --- Uniform Mass ---
                            um = UniformMassCalibration()
                            um.fit(cal_scores_sample, cal_errors_sample)
                            test_um = um.calibrate(test_scores)
                            results.append({
                                **base_entry,
                                "method": "um",
                                "rocauc": float(compute_rocauc(test_um, test_errors)),
                                "mce": float(compute_mce_discrete(test_um, test_errors)),
                            })

                            # --- Platt Scaling ---
                            try:
                                a, b, mu, sigma = fit_platt_general(
                                    cal_scores_sample, cal_errors_sample
                                )
                                test_platt = calibrate_platt_general(
                                    test_scores, a, b, mu, sigma
                                )
                                platt_rocauc = float(compute_rocauc(test_platt, test_errors))
                                platt_mce = float(compute_mce_uniform_mass(
                                    test_platt, test_errors
                                ))
                            except Exception:
                                platt_rocauc = float("nan")
                                platt_mce = float("nan")
                            results.append({
                                **base_entry,
                                "method": "platt",
                                "rocauc": platt_rocauc,
                                "mce": platt_mce,
                            })

                            # --- Isotonic Regression ---
                            try:
                                iso = IsotonicRegression(out_of_bounds="clip")
                                iso.fit(cal_scores_sample, cal_errors_sample)
                                test_iso = iso.predict(test_scores)
                                iso_rocauc = float(compute_rocauc(test_iso, test_errors))
                                iso_mce = float(compute_mce_uniform_mass(
                                    test_iso, test_errors
                                ))
                            except Exception:
                                iso_rocauc = float("nan")
                                iso_mce = float("nan")
                            results.append({
                                **base_entry,
                                "method": "isotonic",
                                "rocauc": iso_rocauc,
                                "mce": iso_mce,
                            })

                print(f"  Done: {len(SCORES)} scores × "
                      f"{len(n_cal_values)} n_cal × {n_draws} draws")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "config": {
            "datasets": DATASETS,
            "models": MODEL_SHORT_NAMES,
            "seeds": SEEDS,
            "scores": SCORES,
            "n_cal_values": n_cal_values,
            "n_draws": n_draws,
            "alpha": ALPHA,
        },
        "theoretical_epsilon": {
            str(k): v for k, v in theoretical_epsilon.items()
        },
        "results": results,
    }

    json_path = os.path.join(output_dir, "ncal_ablation.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} result rows to {json_path}")

    # Print summary per score
    for score_name in SCORES:
        print(f"\n{'='*70}")
        print(f"SUMMARY [{score_name.upper()}]: Mean MCE by method and n_cal")
        print(f"{'='*70}")
        print(f"{'n_cal':>8} {'UM MCE':>10} {'Platt MCE':>10} {'Iso MCE':>10} {'epsilon':>10}")
        print("-" * 50)

        score_results = [r for r in results if r["score"] == score_name]
        for n_cal in n_cal_values:
            rows = [r for r in score_results if r["n_cal"] == n_cal]
            mce_by_method = {}
            for method in ["um", "platt", "isotonic"]:
                method_rows = [r for r in rows if r["method"] == method]
                mces = [r["mce"] for r in method_rows if not math.isnan(r["mce"])]
                mce_by_method[method] = np.mean(mces) if mces else float("nan")
            eps = theoretical_epsilon.get(n_cal, float("nan"))
            print(f"{n_cal:>8} {mce_by_method['um']:>10.4f} "
                  f"{mce_by_method['platt']:>10.4f} "
                  f"{mce_by_method['isotonic']:>10.4f} {eps:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="n_cal ablation for calibration methods"
    )
    parser.add_argument("--cache_dir", type=str, default="cache/paper")
    parser.add_argument("--output_dir", type=str, default="results/ncal_ablation")
    parser.add_argument("--n_draws", type=int, default=20,
                        help="Number of random draws per n_cal value")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    print("n_cal ablation experiment")
    print(f"  Datasets: {DATASETS}")
    print(f"  Models: {MODEL_SHORT_NAMES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Scores: {SCORES}")
    print(f"  n_cal values: {N_CAL_VALUES}")
    print(f"  Draws per n_cal: {args.n_draws}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Output dir: {args.output_dir}")
    print()

    print("Theoretical epsilon values:")
    run_ablation(cache_dir, N_CAL_VALUES, args.n_draws, args.output_dir)


if __name__ == "__main__":
    main()
