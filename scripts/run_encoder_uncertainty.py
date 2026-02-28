#!/usr/bin/env python3
"""
Run uncertainty score calibration for encoder experiments (ELECTRA/MRPC).

This script implements Step 6 of the ELECTRA/MRPC experiment plan:
1. Load cached features, probabilities, and MD scores (from finetune_electra_mrpc.py)
2. Compute all uncertainty scores (MD + softmax-based from class probabilities)
3. For each score x calibration method combination:
   - Fit score calibrator on calibration set
   - Evaluate on test set
4. Report results with proper conventions:
   - MD scores: ROCAUC on raw scores; ECE/BCE only after calibration
   - Softmax-based scores: all metrics before and after calibration
5. Save results to JSON

Usage:
    python run_encoder_uncertainty.py --cache_prefix cache/encoder/electra_mrpc_seed42

    # With specific calibrators and scores
    python run_encoder_uncertainty.py \
        --cache_prefix cache/encoder/electra_mrpc_seed42 \
        --scores mahalanobis_distance max_proba_complement margin \
        --calibrators none uniform_mass

    # Multi-seed run
    for seed in 0 1 2 3 4 5; do
        python run_encoder_uncertainty.py \
            --cache_prefix cache/encoder/electra_mrpc_seed${seed}
    done

Prerequisite: Run finetune_electra_mrpc.py first to generate cached outputs.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json

from src.uncertainty import (
    compute_uncertainty_scores,
    get_predictions_and_errors,
)
from src.score_calibration import get_calibrator
from src.evaluation import (
    compute_rocauc,
    compute_rcc_auc,
    compute_ece,
    compute_ece_uniform_mass,
    compute_ece_discrete,
    compute_binary_cross_entropy,
    compare_calibration,
)
from src.utils import save_results, set_seed, setup_logging


# Scores that are bounded in [0,1] and can have ECE/BCE reported before calibration
BOUNDED_SCORES = {"max_proba_complement", "margin", "doctor", "doctor_normalized",
                  "predictive_entropy"}

# Scores that are unbounded -- ECE/BCE only reported after calibration
UNBOUNDED_SCORES = {"mahalanobis_distance", "energy"}

# PHC methods that require scores in [0,1] (due to logit transform)
PHC_METHODS = {"phc_dp", "phc_ts", "phc_bo"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encoder Uncertainty Score Calibration"
    )

    # Cached data from finetune_electra_mrpc.py
    parser.add_argument("--cache_prefix", type=str, required=True,
                        help="Cache file prefix (e.g., cache/encoder/electra_mrpc_seed42)")

    # Uncertainty scores to evaluate
    parser.add_argument("--scores", type=str, nargs="+",
                        default=[
                            "mahalanobis_distance",
                            "max_proba_complement",
                            "margin",
                            "doctor",
                            "doctor_normalized",
                            "predictive_entropy",
                            "energy",
                        ],
                        help="Uncertainty scores to evaluate")

    # Calibration methods
    parser.add_argument("--calibrators", type=str, nargs="+",
                        default=["none", "uniform_mass"],
                        help="Calibration methods to compare")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="results/encoder_experiments",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_cached_data(cache_prefix: str, logger) -> dict:
    """
    Load cached outputs from finetune_electra_mrpc.py.

    Args:
        cache_prefix: File path prefix (e.g., cache/encoder/electra_mrpc_seed42)
        logger: Logger instance

    Returns:
        Dict with cal and test data (features, labels, probs, predictions, md_scores)
    """
    result = {}
    for split in ["cal", "test"]:
        npz_path = f"{cache_prefix}_{split}.npz"
        logger.info(f"Loading {split} data from {npz_path}")

        data = np.load(npz_path)
        result[split] = {
            "features": data["features"],
            "labels": data["labels"],
            "probs": data["probs"],
            "predictions": data["predictions"],
            "md_scores": data["md_scores"],
        }
        if "logits" in data.files:
            result[split]["logits"] = data["logits"]
        logger.info(f"  {split}: {len(data['labels'])} samples, "
                    f"features={data['features'].shape}, "
                    f"probs={data['probs'].shape}"
                    f"{', logits='+str(data['logits'].shape) if 'logits' in data.files else ''}")

    # Load metadata
    metadata_path = f"{cache_prefix}_metadata.json"
    try:
        with open(metadata_path, "r") as f:
            result["metadata"] = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {metadata_path}")
        result["metadata"] = {}

    return result


def get_score_array(
    score_name: str,
    probs: np.ndarray,
    md_scores: np.ndarray,
    logits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Get a specific uncertainty score array.

    Args:
        score_name: Name of the score
        probs: (N, K) class probabilities (for softmax-based scores)
        md_scores: (N,) pre-computed Mahalanobis distance scores
        logits: (N, K) raw logits, optional. Needed for energy score.

    Returns:
        (N,) array of uncertainty scores
    """
    if score_name == "mahalanobis_distance":
        return md_scores

    if score_name == "energy":
        if logits is None:
            raise ValueError(
                "Energy score requires logits, but none are cached. "
                "Re-run finetune_encoder.py to cache logits."
            )
        from src.uncertainty import compute_energy
        return compute_energy(logits)

    # Softmax-based scores from class probabilities
    scores_obj = compute_uncertainty_scores(probs, logits=logits)
    scores_dict = scores_obj.to_dict()

    if score_name not in scores_dict:
        raise ValueError(f"Unknown score: {score_name}. "
                        f"Available: {list(scores_dict.keys()) + ['mahalanobis_distance']}")

    return scores_dict[score_name]


def run_single_experiment(
    score_name: str,
    calibrator_name: str,
    scores_cal: np.ndarray,
    errors_cal: np.ndarray,
    scores_test: np.ndarray,
    errors_test: np.ndarray,
    logger,
) -> dict:
    """
    Run calibration experiment for a single score x calibrator combination.

    Handles the distinction between bounded and unbounded scores:
    - Bounded scores: report ECE/BCE before and after calibration
    - Unbounded scores (MD): report ECE/BCE only after calibration

    For unbounded scores with PHC methods (which require logit transform on [0,1]),
    the experiment is skipped with a warning.

    Args:
        score_name: Uncertainty score name
        calibrator_name: Calibration method name
        scores_cal: (N_cal,) calibration scores
        errors_cal: (N_cal,) calibration error indicators
        scores_test: (N_test,) test scores
        errors_test: (N_test,) test error indicators
        logger: Logger instance

    Returns:
        Dict with metrics
    """
    is_unbounded = score_name in UNBOUNDED_SCORES
    is_phc = calibrator_name in PHC_METHODS

    # Check compatibility: PHC methods require bounded scores
    if is_unbounded and is_phc:
        logger.warning(
            f"  Skipping {calibrator_name} for {score_name}: "
            f"PHC methods require scores in [0,1], but MD scores are unbounded. "
            f"Platt scaling adaptation for unbounded scores is deferred."
        )
        return {"skipped": True, "reason": "PHC requires bounded scores"}

    # Compute ROCAUC and RCC-AUC on raw scores (works for any score, bounded or not)
    rocauc_before = compute_rocauc(scores_test, errors_test)
    rcc_auc_value = compute_rcc_auc(scores_test, errors_test)

    # Scott's rule for number of bins
    n_bins_scott = int(2 * (len(scores_test) ** (1/3)))

    # Build results dict
    results = {
        "skipped": False,
        "rcc_auc": float(rcc_auc_value),
    }

    # For unbounded scores with 'none' calibrator: NoCalibration clips to [0,1]
    # which destroys unbounded scores. Only report raw ROCAUC, skip ECE/BCE.
    if is_unbounded and calibrator_name == "none":
        results["before"] = {
            "rocauc": float(rocauc_before),
            "ece": None,
            "binary_cross_entropy": None,
        }
        results["after"] = {
            "rocauc": float(rocauc_before),  # Same as before (no calibration)
            "ece": None,
            "binary_cross_entropy": None,
        }
        results["error_rate_cal"] = float(errors_cal.mean())
        results["error_rate_test"] = float(errors_test.mean())
        return results

    # Fit calibrator
    calibrator = get_calibrator(calibrator_name)
    calibrator.fit(scores_cal, errors_cal)

    # Calibrate
    calibrated_cal = calibrator.calibrate(scores_cal)
    calibrated_test = calibrator.calibrate(scores_test)

    # Compute ROCAUC on calibrated scores
    rocauc_after = compute_rocauc(calibrated_test, errors_test)

    # Determine if calibrated output is discrete (binning-based calibrators)
    is_discrete_calibrator = calibrator_name in {"uniform_mass", "quantile", "quantile_binning"}

    if is_unbounded:
        # Unbounded scores: ECE/BCE only after calibration
        if is_discrete_calibrator:
            ece_after = compute_ece_discrete(calibrated_test, errors_test)
        else:
            ece_after = compute_ece_uniform_mass(
                calibrated_test, errors_test, n_bins=n_bins_scott
            )

        results["before"] = {
            "rocauc": float(rocauc_before),
            "ece": None,  # Not meaningful for raw MD scores
            "binary_cross_entropy": None,
        }
        results["after"] = {
            "rocauc": float(rocauc_after),
            "ece": float(ece_after),
            "binary_cross_entropy": float(compute_binary_cross_entropy(
                calibrated_test, errors_test
            )),
        }
    else:
        # Bounded scores: use the standard compare_calibration function
        comparison = compare_calibration(
            scores=scores_test,
            errors=errors_test,
            calibrated_probs=calibrated_test,
            calibrated_discrete=is_discrete_calibrator,
        )
        results["before"] = comparison["before"]
        results["after"] = comparison["after"]

    # Add calibrator info
    if hasattr(calibrator, "get_params"):
        results["calibrator_params"] = calibrator.get_params()
    if hasattr(calibrator, "n_bins_actual"):
        results["n_bins_actual"] = calibrator.n_bins_actual

    # Error rates
    results["error_rate_cal"] = float(errors_cal.mean())
    results["error_rate_test"] = float(errors_test.mean())

    return results


def print_summary_table(all_results: dict, score_names: list,
                        calibrator_names: list, logger) -> None:
    """
    Print formatted summary tables.

    Separate tables for:
    1. ROCAUC (before -> after)
    2. ECE (before -> after, N/A for raw MD)
    3. BCE (before -> after, N/A for raw MD)
    """
    # RCC-AUC table (ranking metric, same for all calibrators)
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: RCC-AUC (lower is better, computed on raw scores)")
    logger.info("=" * 80)

    logger.info(f"{'Score':<25} {'RCC-AUC':<12}")
    logger.info("-" * 40)
    seen_scores = set()
    for score_name in score_names:
        if score_name in seen_scores:
            continue
        # RCC-AUC is the same regardless of calibrator, take from first non-skipped
        for cal in calibrator_names:
            res = all_results["results"].get(score_name, {}).get(cal, {})
            if not res.get("skipped", False) and "rcc_auc" in res:
                logger.info(f"{score_name:<25} {res['rcc_auc']:.2f}")
                seen_scores.add(score_name)
                break

    # ROCAUC table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: ROCAUC (before -> after)")
    logger.info("=" * 80)

    header = f"{'Score':<25}"
    for cal in calibrator_names:
        header += f" {cal:<18}"
    logger.info(header)
    logger.info("-" * 80)

    for score_name in score_names:
        row = f"{score_name:<25}"
        for cal in calibrator_names:
            res = all_results["results"].get(score_name, {}).get(cal, {})
            if res.get("skipped", False):
                row += f" {'SKIPPED':<18}"
            else:
                before = res.get("before", {}).get("rocauc", None)
                after = res.get("after", {}).get("rocauc", None)
                if before is not None and after is not None:
                    row += f" {before:.4f}->{after:.4f} "
                else:
                    row += f" {'N/A':<18}"
        logger.info(row)

    # ECE table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: ECE (before -> after)")
    logger.info("  Note: 'N/A' for raw MD scores (not a probability)")
    logger.info("=" * 80)

    logger.info(header)
    logger.info("-" * 80)

    for score_name in score_names:
        row = f"{score_name:<25}"
        for cal in calibrator_names:
            res = all_results["results"].get(score_name, {}).get(cal, {})
            if res.get("skipped", False):
                row += f" {'SKIPPED':<18}"
            else:
                before = res.get("before", {}).get("ece", None)
                after = res.get("after", {}).get("ece", None)
                if before is not None and after is not None:
                    row += f" {before:.4f}->{after:.4f} "
                elif after is not None:
                    row += f" {'N/A':<6}->{after:.4f} "
                else:
                    row += f" {'N/A':<18}"
        logger.info(row)

    # BCE table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Binary Cross-Entropy (before -> after)")
    logger.info("  Note: 'N/A' for raw MD scores (not a probability)")
    logger.info("=" * 80)

    logger.info(header)
    logger.info("-" * 80)

    for score_name in score_names:
        row = f"{score_name:<25}"
        for cal in calibrator_names:
            res = all_results["results"].get(score_name, {}).get(cal, {})
            if res.get("skipped", False):
                row += f" {'SKIPPED':<18}"
            else:
                before = res.get("before", {}).get("binary_cross_entropy", None)
                after = res.get("after", {}).get("binary_cross_entropy", None)
                if before is not None and after is not None:
                    row += f" {before:.4f}->{after:.4f} "
                elif after is not None:
                    row += f" {'N/A':<6}->{after:.4f} "
                else:
                    row += f" {'N/A':<18}"
        logger.info(row)

    # Error rate
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Classification Error Rate")
    logger.info("=" * 80)

    # Get error rate from any non-skipped result
    for score_name in score_names:
        for cal in calibrator_names:
            res = all_results["results"].get(score_name, {}).get(cal, {})
            if not res.get("skipped", False) and "error_rate_test" in res:
                logger.info(f"  Test error rate: {res['error_rate_test']:.4f}")
                logger.info(f"  Cal error rate:  {res['error_rate_cal']:.4f}")
                return


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Encoder Uncertainty Score Calibration")
    logger.info("=" * 70)
    logger.info(f"Cache prefix: {args.cache_prefix}")
    logger.info(f"Scores: {args.scores}")
    logger.info(f"Calibrators: {args.calibrators}")
    logger.info(f"Output dir: {output_dir}")

    # =========================================================================
    # Load cached data
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Loading cached data")
    logger.info("=" * 70)

    cached = load_cached_data(args.cache_prefix, logger)

    cal_data = cached["cal"]
    test_data = cached["test"]
    metadata = cached["metadata"]

    # Compute error indicators from predictions
    _, errors_cal = get_predictions_and_errors(cal_data["probs"], cal_data["labels"])
    _, errors_test = get_predictions_and_errors(test_data["probs"], test_data["labels"])

    logger.info(f"\nCal error rate: {errors_cal.mean():.4f}")
    logger.info(f"Test error rate: {errors_test.mean():.4f}")

    # =========================================================================
    # Run calibration experiments
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Running calibration experiments")
    logger.info("=" * 70)
    logger.info("Note: Calibrators fitted on ALL calibration data, evaluated on test set")
    logger.info("      Uniform Mass uses Scott's rule for bins: B = 2 * N^(1/3)")

    all_results = {
        "config": {
            "cache_prefix": args.cache_prefix,
            "scores": args.scores,
            "calibrators": args.calibrators,
            "seed": args.seed,
            "metadata": metadata,
        },
        "results": {},
    }

    # Check if logits are available
    cal_logits = cal_data.get("logits")
    test_logits = test_data.get("logits")
    has_logits = cal_logits is not None and test_logits is not None
    if has_logits:
        logger.info("Logits available in cache — energy score enabled")
    else:
        logger.info("No logits in cache — energy score will be skipped if requested")

    for score_name in args.scores:
        # Skip energy if logits are not cached
        if score_name == "energy" and not has_logits:
            logger.info(f"\n--- Score: {score_name} --- SKIPPED (no logits in cache)")
            continue

        logger.info(f"\n--- Score: {score_name} ---")
        all_results["results"][score_name] = {}

        # Get score arrays
        scores_cal = get_score_array(
            score_name, cal_data["probs"], cal_data["md_scores"],
            logits=cal_logits,
        )
        scores_test = get_score_array(
            score_name, test_data["probs"], test_data["md_scores"],
            logits=test_logits,
        )

        for calibrator_name in args.calibrators:
            logger.info(f"  Calibrator: {calibrator_name}")

            results = run_single_experiment(
                score_name=score_name,
                calibrator_name=calibrator_name,
                scores_cal=scores_cal,
                errors_cal=errors_cal,
                scores_test=scores_test,
                errors_test=errors_test,
                logger=logger,
            )

            all_results["results"][score_name][calibrator_name] = results

            # Print summary for this combination
            if not results.get("skipped", False):
                before = results["before"]
                after = results["after"]
                rocauc_str = f"{before['rocauc']:.4f} -> {after['rocauc']:.4f}"
                logger.info(f"    ROCAUC:  {rocauc_str}")
                logger.info(f"    RCC-AUC: {results['rcc_auc']:.2f}")

                ece_before = before.get("ece")
                ece_after = after.get("ece")
                bce_before = before.get("binary_cross_entropy")
                bce_after = after.get("binary_cross_entropy")

                if ece_before is not None and ece_after is not None:
                    logger.info(f"    ECE:    {ece_before:.4f} -> {ece_after:.4f}")
                    logger.info(f"    BCE:    {bce_before:.4f} -> {bce_after:.4f}")
                elif ece_after is not None:
                    logger.info(f"    ECE:    N/A -> {ece_after:.4f}")
                    logger.info(f"    BCE:    N/A -> {bce_after:.4f}")
                else:
                    logger.info(f"    ECE:    N/A (no calibration applied)")
                    logger.info(f"    BCE:    N/A (no calibration applied)")

    # =========================================================================
    # Save results
    # =========================================================================
    # Derive output filename from cache prefix
    cache_name = Path(args.cache_prefix).name
    results_file = output_dir / f"{cache_name}_uncertainty_calibration.json"
    save_results(all_results, str(results_file))
    logger.info(f"\nResults saved to {results_file}")

    # =========================================================================
    # Print summary tables
    # =========================================================================
    print_summary_table(all_results, args.scores, args.calibrators, logger)

    logger.info("\n" + "=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
