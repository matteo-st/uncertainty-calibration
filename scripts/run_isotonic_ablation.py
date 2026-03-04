#!/usr/bin/env python3
"""
Isotonic regression interpolation ablation: linear vs step across n_cal values.

For each n_cal, samples calibration sets from the pool, fits isotonic regression,
and evaluates test set using both linear interpolation (sklearn default) and
step function (piecewise-constant PAV output).

Reports ROCAUC, ECE, MCE for both modes.

Usage:
    python scripts/run_isotonic_ablation.py \
        --cache_dir cache/paper \
        --output_dir results/isotonic_ablation \
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
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from src.evaluation import (
    compute_rocauc,
    compute_ece_discrete,
    compute_ece_uniform_mass,
    compute_mce_discrete,
    compute_mce_uniform_mass,
)

# --- Configuration ---

DATASETS = ["sst2", "agnews"]
SEEDS = [42, 123, 456]
N_CAL_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
SCORES = ["sp", "md"]
MODEL = "deberta"


def extract_scores(data, score_name):
    if score_name == "sp":
        return 1.0 - np.max(data["probs"], axis=1)
    elif score_name == "md":
        return data["md_scores"]
    raise ValueError(f"Unknown score: {score_name}")


def load_npz(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


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


def get_n_breakpoints(ir):
    if hasattr(ir, 'X_thresholds_'):
        return len(ir.X_thresholds_)
    elif hasattr(ir, 'f_'):
        return len(ir.f_.x)
    return -1


def run_ablation(cache_dir, n_cal_values, n_draws, output_dir):
    results = []
    total_combos = len(DATASETS) * len(SEEDS)
    combo_idx = 0

    for dataset in DATASETS:
        for seed in SEEDS:
            combo_idx += 1
            variant = f"{MODEL}_{dataset}"
            prefix = str(cache_dir / f"{variant}_seed{seed}")

            print(f"\n[{combo_idx}/{total_combos}] {variant} seed={seed}")

            cal_path = f"{prefix}_cal.npz"
            unused_path = f"{prefix}_unused.npz"
            test_path = f"{prefix}_test.npz"

            if not Path(unused_path).exists():
                print(f"  SKIP: {unused_path} not found")
                continue

            cal_data = load_npz(cal_path)
            unused_data = load_npz(unused_path)
            test_data = load_npz(test_path)

            pool_errors = np.concatenate([
                (cal_data["predictions"] != cal_data["labels"]).astype(float),
                (unused_data["predictions"] != unused_data["labels"]).astype(float),
            ])
            test_errors = (
                test_data["predictions"] != test_data["labels"]
            ).astype(float)

            for score_name in SCORES:
                pool_scores = np.concatenate([
                    extract_scores(cal_data, score_name),
                    extract_scores(unused_data, score_name),
                ])
                test_scores = extract_scores(test_data, score_name)
                pool_size = len(pool_scores)

                raw_rocauc = compute_rocauc(test_scores, test_errors)
                print(f"  [{score_name.upper()}] Pool: {pool_size}, "
                      f"test: {len(test_scores)}, raw ROCAUC: {raw_rocauc:.4f}")

                for n_cal in n_cal_values:
                    if n_cal > pool_size:
                        print(f"    SKIP n_cal={n_cal} > pool_size={pool_size}")
                        continue

                    for draw in range(n_draws):
                        rng = np.random.RandomState(seed * 1000 + draw)
                        idx = rng.choice(pool_size, n_cal, replace=False)
                        cal_scores_sample = pool_scores[idx]
                        cal_errors_sample = pool_errors[idx]

                        if len(np.unique(cal_errors_sample)) < 2:
                            continue

                        base_entry = {
                            "dataset": dataset,
                            "seed": seed,
                            "score": score_name,
                            "n_cal": n_cal,
                            "draw": draw,
                            "raw_rocauc": float(raw_rocauc),
                        }

                        try:
                            ir = IsotonicRegression(out_of_bounds="clip")
                            ir.fit(cal_scores_sample, cal_errors_sample)
                            n_bp = get_n_breakpoints(ir)

                            # Linear interpolation (sklearn default)
                            pred_linear = ir.predict(test_scores)
                            results.append({
                                **base_entry,
                                "method": "isotonic_linear",
                                "rocauc": float(compute_rocauc(pred_linear, test_errors)),
                                "mce": float(compute_mce_uniform_mass(pred_linear, test_errors)),
                                "ece": float(compute_ece_uniform_mass(pred_linear, test_errors)),
                                "n_unique": int(len(np.unique(pred_linear))),
                                "n_breakpoints": n_bp,
                            })

                            # Step function
                            pred_step = isotonic_predict_step(ir, test_scores)
                            results.append({
                                **base_entry,
                                "method": "isotonic_step",
                                "rocauc": float(compute_rocauc(pred_step, test_errors)),
                                "mce": float(compute_mce_uniform_mass(pred_step, test_errors)),
                                "ece": float(compute_ece_uniform_mass(pred_step, test_errors)),
                                "n_unique": int(len(np.unique(pred_step))),
                                "n_breakpoints": n_bp,
                            })
                        except Exception as e:
                            print(f"    ERROR n_cal={n_cal} draw={draw}: {e}")

            print(f"  Done: {len(SCORES)} scores × "
                  f"{len(n_cal_values)} n_cal × {n_draws} draws")

    os.makedirs(output_dir, exist_ok=True)
    output = {
        "config": {
            "datasets": DATASETS,
            "model": MODEL,
            "seeds": SEEDS,
            "scores": SCORES,
            "n_cal_values": n_cal_values,
            "n_draws": n_draws,
        },
        "results": results,
    }

    json_path = os.path.join(output_dir, "isotonic_ablation.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} result rows to {json_path}")

    # Print summary
    for score_name in SCORES:
        for metric in ["rocauc", "mce", "ece"]:
            print(f"\n{'='*70}")
            print(f"[{score_name.upper()}] Mean {metric.upper()} by method and n_cal")
            print(f"{'='*70}")
            print(f"{'n_cal':>8} {'Linear':>12} {'Step':>12} {'Diff':>10}")
            print("-" * 50)

            score_results = [r for r in results if r["score"] == score_name]
            for n_cal in n_cal_values:
                rows = [r for r in score_results if r["n_cal"] == n_cal]
                vals = {}
                for method in ["isotonic_linear", "isotonic_step"]:
                    method_rows = [r for r in rows if r["method"] == method]
                    v = [r[metric] for r in method_rows if not math.isnan(r[metric])]
                    vals[method] = np.mean(v) if v else float("nan")
                diff = vals["isotonic_linear"] - vals["isotonic_step"]
                print(f"{n_cal:>8} {vals['isotonic_linear']:>12.4f} "
                      f"{vals['isotonic_step']:>12.4f} {diff:>+10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Isotonic regression interpolation ablation"
    )
    parser.add_argument("--cache_dir", type=str, default="cache/paper")
    parser.add_argument("--output_dir", type=str, default="results/isotonic_ablation")
    parser.add_argument("--n_draws", type=int, default=20)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print("Isotonic interpolation ablation")
    print(f"  Datasets: {DATASETS}")
    print(f"  Model: {MODEL}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Scores: {SCORES}")
    print(f"  n_cal values: {N_CAL_VALUES}")
    print(f"  Draws per n_cal: {args.n_draws}")

    run_ablation(cache_dir, N_CAL_VALUES, args.n_draws, args.output_dir)


if __name__ == "__main__":
    main()
