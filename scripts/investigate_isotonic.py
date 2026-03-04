"""
Investigate isotonic regression interpolation: linear (sklearn default) vs step function.

For each (dataset, score, seed), fits sklearn IsotonicRegression on the calibration set,
then evaluates on the test set using two prediction modes:
  1. Linear interpolation (sklearn default): continuous output
  2. Step function (kind='previous'): piecewise-constant output (the "true" PAV result)

Reports ROCAUC, ECE, MCE for both modes using uniform-mass binning,
plus level-set ECE/MCE for the step function.

Usage:
    python scripts/investigate_isotonic.py --cache_dir cache/paper --model deberta
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    compute_rocauc,
    compute_ece_uniform_mass,
    compute_mce_uniform_mass,
    compute_ece_discrete,
    compute_mce_discrete,
)
from src.uncertainty import compute_max_proba_complement


DATASETS = ["mrpc", "sst2", "cola", "agnews"]
SEEDS = [42, 123, 456]
SCORES = ["sp", "md"]


def load_split(prefix, split):
    path = f"{prefix}_{split}.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_scores(data, score_key):
    if score_key == "sp":
        return compute_max_proba_complement(data["probs"])
    elif score_key == "md":
        return data["md_scores"]
    raise ValueError(f"Unknown score: {score_key}")


def isotonic_predict_step(ir, X):
    """Predict using step function (piecewise constant) instead of linear interpolation.

    Uses the PAV breakpoints stored in ir.X_thresholds_ / ir.y_thresholds_
    (sklearn >= 1.1) or ir.f_.x / ir.f_.y (earlier).
    """
    # Get breakpoints
    if hasattr(ir, 'X_thresholds_') and hasattr(ir, 'y_thresholds_'):
        x_bp = ir.X_thresholds_
        y_bp = ir.y_thresholds_
    elif hasattr(ir, 'f_'):
        x_bp = ir.f_.x
        y_bp = ir.f_.y
    else:
        # Fallback: get from X_min_, X_max_ and internal state
        raise RuntimeError("Cannot extract breakpoints from IsotonicRegression")

    # Build step function using 'previous' interpolation (left-continuous step)
    if len(x_bp) == 1:
        return np.full_like(X, y_bp[0], dtype=float)

    f_step = interp1d(
        x_bp, y_bp, kind='previous',
        bounds_error=False,
        fill_value=(y_bp[0], y_bp[-1])
    )
    result = f_step(X)

    # Handle points exactly at x_bp[-1] (interp1d 'previous' may not include right edge)
    result[X >= x_bp[-1]] = y_bp[-1]

    return result


def evaluate_isotonic(cal_scores, cal_errors, test_scores, test_errors):
    """Fit isotonic regression and compare linear vs step prediction."""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(cal_scores, cal_errors)

    # --- Linear interpolation (sklearn default) ---
    pred_linear = ir.predict(test_scores)

    # --- Step function ---
    pred_step = isotonic_predict_step(ir, test_scores)

    # Number of breakpoints
    if hasattr(ir, 'X_thresholds_'):
        n_breakpoints = len(ir.X_thresholds_)
    elif hasattr(ir, 'f_'):
        n_breakpoints = len(ir.f_.x)
    else:
        n_breakpoints = -1

    # Metrics for linear
    linear_metrics = {
        "rocauc": float(compute_rocauc(pred_linear, test_errors)),
        "ece_um": float(compute_ece_uniform_mass(pred_linear, test_errors)),
        "mce_um": float(compute_mce_uniform_mass(pred_linear, test_errors)),
        "n_unique": int(len(np.unique(pred_linear))),
    }

    # Metrics for step
    step_metrics = {
        "rocauc": float(compute_rocauc(pred_step, test_errors)),
        "ece_um": float(compute_ece_uniform_mass(pred_step, test_errors)),
        "mce_um": float(compute_mce_uniform_mass(pred_step, test_errors)),
        # Level-set metrics (natural for discrete output)
        "ece_ls": float(compute_ece_discrete(pred_step, test_errors)),
        "mce_ls": float(compute_mce_discrete(pred_step, test_errors)),
        "n_unique": int(len(np.unique(pred_step))),
    }

    # Also compute level-set metrics for linear (for comparison, even if many groups)
    linear_metrics["ece_ls"] = float(compute_ece_discrete(pred_linear, test_errors))
    linear_metrics["mce_ls"] = float(compute_mce_discrete(pred_linear, test_errors))

    return {
        "linear": linear_metrics,
        "step": step_metrics,
        "n_breakpoints": n_breakpoints,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cache/paper")
    parser.add_argument("--model", default="deberta")
    parser.add_argument("--output_dir", default="results/isotonic_investigation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for dataset in DATASETS:
        for seed in SEEDS:
            prefix = os.path.join(args.cache_dir, f"{args.model}_{dataset}_seed{seed}")
            exp_name = f"{args.model}_{dataset}_seed{seed}"

            try:
                cal_data = load_split(prefix, "cal")
                test_data = load_split(prefix, "test")
            except FileNotFoundError as e:
                print(f"SKIP {exp_name}: {e}")
                continue

            cal_errors = (cal_data["predictions"] != cal_data["labels"]).astype(float)
            test_errors = (test_data["predictions"] != test_data["labels"]).astype(float)

            exp_results = {
                "n_cal": int(len(cal_errors)),
                "n_test": int(len(test_errors)),
                "error_rate": float(test_errors.mean()),
            }

            for score_key in SCORES:
                try:
                    cal_scores = get_scores(cal_data, score_key)
                    test_scores = get_scores(test_data, score_key)
                except (KeyError, ValueError) as e:
                    print(f"  SKIP {exp_name}/{score_key}: {e}")
                    continue

                result = evaluate_isotonic(
                    cal_scores, cal_errors, test_scores, test_errors
                )
                exp_results[score_key] = result

                # Print summary
                lin = result["linear"]
                stp = result["step"]
                print(f"{exp_name:>30s} | {score_key:>3s} | "
                      f"BP={result['n_breakpoints']:>3d} | "
                      f"Linear: ROCAUC={lin['rocauc']:.4f} ECE_um={lin['ece_um']:.4f} MCE_um={lin['mce_um']:.4f} unique={lin['n_unique']:>5d} | "
                      f"Step: ROCAUC={stp['rocauc']:.4f} ECE_um={stp['ece_um']:.4f} MCE_um={stp['mce_um']:.4f} unique={stp['n_unique']:>4d}")

            all_results[exp_name] = exp_results

    # Save JSON
    json_path = os.path.join(args.output_dir, "isotonic_investigation.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # --- Aggregate summary ---
    print(f"\n{'='*100}")
    print("AGGREGATE SUMMARY (mean ± std over seeds)")
    print(f"{'='*100}")
    print(f"{'Dataset':<10} {'Score':<5} {'Mode':<8} {'ROCAUC':>8} {'ECE_um':>8} {'MCE_um':>8} {'ECE_ls':>8} {'MCE_ls':>8} {'Unique':>8}")
    print("-" * 85)

    for dataset in DATASETS:
        for score_key in SCORES:
            for mode in ["linear", "step"]:
                vals = {"rocauc": [], "ece_um": [], "mce_um": [], "ece_ls": [], "mce_ls": [], "n_unique": []}
                for seed in SEEDS:
                    exp_name = f"{args.model}_{dataset}_seed{seed}"
                    if exp_name not in all_results or score_key not in all_results[exp_name]:
                        continue
                    m = all_results[exp_name][score_key][mode]
                    for k in vals:
                        vals[k].append(m.get(k, float('nan')))

                if not vals["rocauc"]:
                    continue

                means = {k: np.mean(v) for k, v in vals.items()}
                stds = {k: np.std(v) for k, v in vals.items()}

                print(f"{dataset:<10} {score_key:<5} {mode:<8} "
                      f"{means['rocauc']:.4f}±{stds['rocauc']:.3f} "
                      f"{means['ece_um']:.4f}±{stds['ece_um']:.3f} "
                      f"{means['mce_um']:.4f}±{stds['mce_um']:.3f} "
                      f"{means['ece_ls']:.4f}±{stds['ece_ls']:.3f} "
                      f"{means['mce_ls']:.4f}±{stds['mce_ls']:.3f} "
                      f"{means['n_unique']:>6.0f}±{stds['n_unique']:.0f}")
        print()


if __name__ == "__main__":
    main()
