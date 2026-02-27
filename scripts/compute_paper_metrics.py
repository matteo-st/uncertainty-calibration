"""
Compute paper experiment metrics (ROCAUC, ECE, MCE) for Max Prob and Mahalanobis Distance.

For each experiment (cache prefix), loads cal + test npz files, computes metrics
before and after Uniform Mass calibration, saves results as JSON and test scores as npz.

Usage:
    python scripts/compute_paper_metrics.py \
        --cache_prefixes cache/encoder/electra_mrpc_seed42 \
                         cache/encoder_10pct/electra_sst2_seed42 \
        --output_dir results/paper_metrics
"""

import argparse
import json
import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    compute_rocauc,
    compute_ece_uniform_mass,
    compute_ece_discrete,
    compute_mce_uniform_mass,
    compute_mce_discrete,
)
from src.score_calibration import UniformMassCalibration


def load_split(prefix: str, split: str) -> dict:
    """Load an npz file for a given prefix and split."""
    path = f"{prefix}_{split}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_experiment_metrics(cache_prefix: str) -> dict:
    """
    Compute all paper metrics for one experiment.

    Returns dict with metrics for Max Prob and MD, before and after UM calibration,
    plus the raw/calibrated test scores.
    """
    # Load data
    cal_data = load_split(cache_prefix, "cal")
    test_data = load_split(cache_prefix, "test")

    # Errors
    cal_errors = (cal_data["predictions"] != cal_data["labels"]).astype(float)
    test_errors = (test_data["predictions"] != test_data["labels"]).astype(float)

    # Uncertainty scores
    cal_maxprob = 1.0 - cal_data["probs"].max(axis=1)
    test_maxprob = 1.0 - test_data["probs"].max(axis=1)

    cal_md = cal_data["md_scores"]
    test_md = test_data["md_scores"]

    results = {}

    # --- Max Prob ---
    # Before UM calibration
    results["maxprob_before"] = {
        "rocauc": compute_rocauc(test_maxprob, test_errors),
        "ece": compute_ece_uniform_mass(test_maxprob, test_errors),
        "mce": compute_mce_uniform_mass(test_maxprob, test_errors),
    }

    # After UM calibration
    um_maxprob = UniformMassCalibration()
    um_maxprob.fit(cal_maxprob, cal_errors)
    test_maxprob_cal = um_maxprob.calibrate(test_maxprob)

    results["maxprob_after"] = {
        "rocauc": compute_rocauc(test_maxprob_cal, test_errors),
        "ece": compute_ece_discrete(test_maxprob_cal, test_errors),
        "mce": compute_mce_discrete(test_maxprob_cal, test_errors),
    }

    # --- Mahalanobis Distance ---
    # Before UM calibration (ROCAUC only)
    results["md_before"] = {
        "rocauc": compute_rocauc(test_md, test_errors),
    }

    # After UM calibration
    um_md = UniformMassCalibration()
    um_md.fit(cal_md, cal_errors)
    test_md_cal = um_md.calibrate(test_md)

    results["md_after"] = {
        "rocauc": compute_rocauc(test_md_cal, test_errors),
        "ece": compute_ece_discrete(test_md_cal, test_errors),
        "mce": compute_mce_discrete(test_md_cal, test_errors),
    }

    # Collect test scores for saving
    test_scores = {
        "maxprob_raw": test_maxprob,
        "maxprob_calibrated": test_maxprob_cal,
        "md_raw": test_md,
        "md_calibrated": test_md_cal,
        "errors": test_errors,
        "labels": test_data["labels"],
        "predictions": test_data["predictions"],
    }

    # Metadata
    n_test = len(test_errors)
    n_cal = len(cal_errors)
    error_rate = test_errors.mean()
    um_maxprob_params = um_maxprob.get_params()
    um_md_params = um_md.get_params()

    meta = {
        "n_test": int(n_test),
        "n_cal": int(n_cal),
        "error_rate": float(error_rate),
        "um_maxprob_n_bins": um_maxprob_params["n_bins_actual"],
        "um_md_n_bins": um_md_params["n_bins_actual"],
    }

    return results, test_scores, meta


def format_table(all_results: dict) -> str:
    """Format results as a readable table."""
    lines = []

    header = (
        f"{'Experiment':<35} {'Score':<10} {'Stage':<8} "
        f"{'ROCAUC':>8} {'ECE':>10} {'MCE':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name, (results, _, meta) in all_results.items():
        short_name = os.path.basename(name)
        err_pct = meta["error_rate"] * 100

        # Max Prob before
        r = results["maxprob_before"]
        lines.append(
            f"{short_name:<35} {'MaxProb':<10} {'raw':<8} "
            f"{r['rocauc']:>8.4f} {r['ece']:>10.4f} {r['mce']:>10.4f}"
        )

        # Max Prob after
        r = results["maxprob_after"]
        lines.append(
            f"{'':<35} {'':<10} {'UM-cal':<8} "
            f"{r['rocauc']:>8.4f} {r['ece']:>10.4f} {r['mce']:>10.4f}"
        )

        # MD before
        r = results["md_before"]
        lines.append(
            f"{'':<35} {'MD':<10} {'raw':<8} "
            f"{r['rocauc']:>8.4f} {'---':>10} {'---':>10}"
        )

        # MD after
        r = results["md_after"]
        lines.append(
            f"{'':<35} {'':<10} {'UM-cal':<8} "
            f"{r['rocauc']:>8.4f} {r['ece']:>10.4f} {r['mce']:>10.4f}"
        )

        lines.append(f"  (n_test={meta['n_test']}, error_rate={err_pct:.1f}%, "
                      f"UM bins: maxprob={meta['um_maxprob_n_bins']}, md={meta['um_md_n_bins']})")
        lines.append("")

    return "\n".join(lines)


def run_sanity_checks(results: dict, name: str):
    """Run verification checks on results."""
    checks_passed = True

    # MCE >= ECE (max >= weighted average)
    for key in ["maxprob_before", "maxprob_after", "md_after"]:
        if "ece" in results[key] and "mce" in results[key]:
            ece = results[key]["ece"]
            mce = results[key]["mce"]
            if mce < ece - 1e-10:
                print(f"  WARNING [{name}] {key}: MCE ({mce:.6f}) < ECE ({ece:.6f})")
                checks_passed = False

    # ECE after <= ECE before for MaxProb (calibration should help)
    if "ece" in results["maxprob_before"] and "ece" in results["maxprob_after"]:
        ece_before = results["maxprob_before"]["ece"]
        ece_after = results["maxprob_after"]["ece"]
        if ece_after > ece_before + 0.01:
            print(f"  WARNING [{name}] MaxProb ECE increased after calibration: "
                  f"{ece_before:.4f} -> {ece_after:.4f}")

    if checks_passed:
        print(f"  Sanity checks passed for {name}")


def main():
    parser = argparse.ArgumentParser(description="Compute paper experiment metrics")
    parser.add_argument(
        "--cache_prefixes", nargs="+", required=True,
        help="Cache prefixes for each experiment (e.g. cache/encoder/electra_mrpc_seed42)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/paper_metrics",
        help="Directory to save results"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for prefix in args.cache_prefixes:
        exp_name = os.path.basename(prefix)
        print(f"\n{'='*60}")
        print(f"Processing: {prefix}")
        print(f"{'='*60}")

        results, test_scores, meta = compute_experiment_metrics(prefix)
        all_results[prefix] = (results, test_scores, meta)

        # Sanity checks
        run_sanity_checks(results, exp_name)

        # Save test scores as npz
        scores_path = os.path.join(args.output_dir, f"{exp_name}_test_scores.npz")
        np.savez(scores_path, **test_scores)
        print(f"  Saved test scores: {scores_path}")

    # Print formatted table
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    print(format_table(all_results))

    # Save JSON results (convert numpy types)
    json_results = {}
    for prefix, (results, _, meta) in all_results.items():
        exp_name = os.path.basename(prefix)
        json_results[exp_name] = {
            "metrics": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            "meta": meta,
        }

    json_path = os.path.join(args.output_dir, "paper_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved JSON results: {json_path}")


if __name__ == "__main__":
    main()
