"""
Compute paper experiment metrics (ROCAUC, ECE, MCE) for all uncertainty scores.

Scores: SP (MaxProb), Predictive Entropy, Doctor, Energy, MD, RDE.
For each experiment (cache prefix), loads cal + test npz files, computes metrics
before and after Uniform Mass calibration, saves results as JSON and test scores as npz.

Usage:
    python scripts/compute_paper_metrics.py \
        --cache_prefixes cache/paper/electra_sst2_seed42 \
                         cache/paper/electra_mrpc_seed42 \
        --output_dir results/paper_metrics

    # All seeds for one model/dataset:
    python scripts/compute_paper_metrics.py \
        --cache_prefixes cache/paper/electra_mrpc_seed42 \
                         cache/paper/electra_mrpc_seed123 \
                         cache/paper/electra_mrpc_seed456
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
from src.uncertainty import (
    compute_max_proba_complement,
    compute_predictive_entropy,
    compute_doctor,
    compute_energy,
)


# --- Score definitions ---
# Each score has: key, display name, bounded (bool), compute function signature.
# Bounded scores get ECE/MCE before UM. Unbounded scores get ROCAUC only before UM.
SCORE_DEFS = [
    {
        "key": "sp",
        "name": "SP",
        "bounded": True,
        "source": "probs",  # computed from softmax probs
    },
    {
        "key": "pe",
        "name": "PE",
        "bounded": True,
        "source": "probs",
    },
    {
        "key": "doctor",
        "name": "Doctor",
        "bounded": True,
        "source": "probs",
    },
    {
        "key": "energy",
        "name": "Energy",
        "bounded": False,
        "source": "logits",  # computed from raw logits
    },
    {
        "key": "md",
        "name": "MD",
        "bounded": False,
        "source": "cache",  # pre-computed in npz
    },
    {
        "key": "rde",
        "name": "RDE",
        "bounded": False,
        "source": "cache",  # pre-computed in npz (when available)
    },
]


def compute_score(score_def: dict, data: dict) -> np.ndarray:
    """Compute a score array from cached data."""
    key = score_def["key"]
    source = score_def["source"]

    if source == "probs":
        probs = data["probs"]
        if key == "sp":
            return compute_max_proba_complement(probs)
        elif key == "pe":
            return compute_predictive_entropy(probs)
        elif key == "doctor":
            return compute_doctor(probs)
    elif source == "logits":
        if key == "energy":
            if "logits" not in data:
                return None
            return compute_energy(data["logits"])
    elif source == "cache":
        if key == "md":
            return data.get("md_scores")
        elif key == "rde":
            return data.get("rde_scores")

    raise ValueError(f"Unknown score: {key}")


def load_split(prefix: str, split: str) -> dict:
    """Load an npz file for a given prefix and split."""
    path = f"{prefix}_{split}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_experiment_metrics(cache_prefix: str, score_defs: list) -> dict:
    """
    Compute all paper metrics for one experiment across all scores.

    Returns:
        results: dict of {score_key: {"before": {...}, "after": {...}}}
        test_scores: dict of raw/calibrated arrays for saving
        meta: dict of metadata
    """
    # Load data
    cal_data = load_split(cache_prefix, "cal")
    test_data = load_split(cache_prefix, "test")

    # Errors
    cal_errors = (cal_data["predictions"] != cal_data["labels"]).astype(float)
    test_errors = (test_data["predictions"] != test_data["labels"]).astype(float)

    results = {}
    test_scores = {
        "errors": test_errors,
        "labels": test_data["labels"],
        "predictions": test_data["predictions"],
    }
    um_bins = {}

    for sdef in score_defs:
        key = sdef["key"]
        bounded = sdef["bounded"]
        name = sdef["name"]

        # Compute scores
        cal_scores = compute_score(sdef, cal_data)
        test_scores_arr = compute_score(sdef, test_data)

        # Skip if score is not available (e.g., no logits for energy, no rde_scores)
        if cal_scores is None or test_scores_arr is None:
            print(f"  Skipping {name}: not available in cache")
            continue

        # --- Before UM ---
        before = {"rocauc": compute_rocauc(test_scores_arr, test_errors)}
        if bounded:
            before["ece"] = compute_ece_uniform_mass(test_scores_arr, test_errors)
            before["mce"] = compute_mce_uniform_mass(test_scores_arr, test_errors)

        results[f"{key}_before"] = before

        # --- After UM calibration ---
        um = UniformMassCalibration()
        um.fit(cal_scores, cal_errors)
        test_cal = um.calibrate(test_scores_arr)

        results[f"{key}_after"] = {
            "rocauc": compute_rocauc(test_cal, test_errors),
            "ece": compute_ece_discrete(test_cal, test_errors),
            "mce": compute_mce_discrete(test_cal, test_errors),
        }

        # Save scores
        test_scores[f"{key}_raw"] = test_scores_arr
        test_scores[f"{key}_calibrated"] = test_cal

        um_params = um.get_params()
        um_bins[key] = um_params["n_bins_actual"]

    # Metadata
    meta = {
        "n_test": int(len(test_errors)),
        "n_cal": int(len(cal_errors)),
        "error_rate": float(test_errors.mean()),
        "um_bins": um_bins,
        "has_logits": "logits" in test_data,
    }

    return results, test_scores, meta


def format_table(all_results: dict, score_defs: list) -> str:
    """Format results as a readable table."""
    lines = []

    header = (
        f"{'Experiment':<35} {'Score':<10} {'Stage':<8} "
        f"{'ROCAUC':>8} {'ECE':>10} {'MCE':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for prefix, (results, _, meta) in all_results.items():
        short_name = os.path.basename(prefix)
        err_pct = meta["error_rate"] * 100
        first_score = True

        for sdef in score_defs:
            key = sdef["key"]
            name = sdef["name"]

            if f"{key}_before" not in results:
                continue

            # Before UM
            r = results[f"{key}_before"]
            exp_col = short_name if first_score else ""
            ece_str = f"{r['ece']:>10.4f}" if "ece" in r else f"{'---':>10}"
            mce_str = f"{r['mce']:>10.4f}" if "mce" in r else f"{'---':>10}"
            lines.append(
                f"{exp_col:<35} {name:<10} {'raw':<8} "
                f"{r['rocauc']:>8.4f} {ece_str} {mce_str}"
            )
            first_score = False

            # After UM
            r = results[f"{key}_after"]
            lines.append(
                f"{'':<35} {'':<10} {'UM-cal':<8} "
                f"{r['rocauc']:>8.4f} {r['ece']:>10.4f} {r['mce']:>10.4f}"
            )

        # Summary line
        bins_str = ", ".join(f"{k}={v}" for k, v in meta["um_bins"].items())
        lines.append(
            f"  (n_test={meta['n_test']}, error_rate={err_pct:.1f}%, "
            f"UM bins: {bins_str})"
        )
        lines.append("")

    return "\n".join(lines)


def run_sanity_checks(results: dict, name: str, score_defs: list):
    """Run verification checks on results."""
    checks_passed = True

    for sdef in score_defs:
        key = sdef["key"]

        # MCE >= ECE for all stages that have both
        for stage in [f"{key}_before", f"{key}_after"]:
            if stage not in results:
                continue
            r = results[stage]
            if "ece" in r and "mce" in r:
                if r["mce"] < r["ece"] - 1e-10:
                    print(f"  WARNING [{name}] {stage}: "
                          f"MCE ({r['mce']:.6f}) < ECE ({r['ece']:.6f})")
                    checks_passed = False

        # For bounded scores: ECE should not increase much after calibration
        if sdef["bounded"] and f"{key}_before" in results and f"{key}_after" in results:
            ece_before = results[f"{key}_before"].get("ece")
            ece_after = results[f"{key}_after"].get("ece")
            if ece_before is not None and ece_after is not None:
                if ece_after > ece_before + 0.01:
                    print(f"  WARNING [{name}] {sdef['name']} ECE increased: "
                          f"{ece_before:.4f} -> {ece_after:.4f}")

        # ROCAUC should be preserved after UM (rank-preserving)
        if f"{key}_before" in results and f"{key}_after" in results:
            roc_before = results[f"{key}_before"]["rocauc"]
            roc_after = results[f"{key}_after"]["rocauc"]
            if abs(roc_before - roc_after) > 0.001:
                print(f"  WARNING [{name}] {sdef['name']} ROCAUC changed: "
                      f"{roc_before:.4f} -> {roc_after:.4f}")

    if checks_passed:
        print(f"  Sanity checks passed for {name}")


def main():
    parser = argparse.ArgumentParser(description="Compute paper experiment metrics")
    parser.add_argument(
        "--cache_prefixes", nargs="+", required=True,
        help="Cache prefixes (e.g. cache/paper/electra_mrpc_seed42)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/paper_metrics",
        help="Directory to save results"
    )
    parser.add_argument(
        "--scores", nargs="+", default=None,
        help="Subset of scores to compute (default: all available). "
             "Keys: sp, pe, doctor, energy, md, rde"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter score definitions if requested
    if args.scores:
        score_defs = [s for s in SCORE_DEFS if s["key"] in args.scores]
    else:
        score_defs = SCORE_DEFS

    all_results = {}

    for prefix in args.cache_prefixes:
        exp_name = os.path.basename(prefix)
        print(f"\n{'='*60}")
        print(f"Processing: {prefix}")
        print(f"{'='*60}")

        results, test_scores, meta = compute_experiment_metrics(
            prefix, score_defs
        )
        all_results[prefix] = (results, test_scores, meta)

        # Sanity checks
        run_sanity_checks(results, exp_name, score_defs)

        # Save test scores as npz
        scores_path = os.path.join(args.output_dir, f"{exp_name}_test_scores.npz")
        np.savez(scores_path, **test_scores)
        print(f"  Saved test scores: {scores_path}")

    # Print formatted table
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    print(format_table(all_results, score_defs))

    # Save JSON results (convert numpy types)
    json_results = {}
    for prefix, (results, _, meta) in all_results.items():
        exp_name = os.path.basename(prefix)
        json_metrics = {}
        for k, v in results.items():
            json_metrics[k] = {kk: float(vv) for kk, vv in v.items()}
        json_results[exp_name] = {
            "metrics": json_metrics,
            "meta": meta,
        }

    json_path = os.path.join(args.output_dir, "paper_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved JSON results: {json_path}")


if __name__ == "__main__":
    main()
