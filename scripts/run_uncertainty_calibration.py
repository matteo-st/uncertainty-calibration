#!/usr/bin/env python3
"""
Run uncertainty score calibration experiments for error prediction.

This script (Steps 2-4 of the experimental pipeline):
1. Loads the LLM and optionally an affine calibrator checkpoint
   OR loads cached raw probabilities (skipping LLM inference)
2. Extracts class probabilities on calibration samples (cached for reuse)
3. Applies affine calibration if checkpoint provided
4. Computes uncertainty scores (1-max_proba, margin, doctor)
5. Fits score calibrators on ALL calibration data (PHC-DP, PHC-TS, PHC-BO, Uniform Mass)
6. Evaluates calibration quality (ROCAUC, ECE, Cross-Entropy)

Prerequisite: Run finetune_affine.py first to get calibrator checkpoint.

Caching: Raw probabilities are cached to avoid re-running LLM inference.
         Affine calibration is applied post-hoc on cached probabilities.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import pickle
from tqdm import tqdm

from src.models import LLMClassifier
from src.data import load_dataset_by_name
from src.calibration import AffineCalibrator
from src.uncertainty import (
    compute_uncertainty_scores,
    get_predictions_and_errors,
)
from src.score_calibration import (
    get_calibrator,
    PlattScaling,
    UniformMassCalibration,
    NoCalibration,
)
from src.evaluation import (
    compute_rocauc,
    compute_ece,
    compute_binary_cross_entropy,
    compare_calibration,
)
from src.utils import save_results, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Uncertainty Score Calibration")

    # Model and data
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--model", type=str, default="gpt2-xl")
    parser.add_argument("--calibrator_checkpoint", type=str, default=None,
                        help="Path to affine calibrator checkpoint (.pkl from finetune_affine.py)")

    # Data sizes
    parser.add_argument("--n_calibration", type=int, default=1000,
                        help="Number of samples for calibration (ALL used for fitting)")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Number of samples for final testing")

    # Few-shot settings
    parser.add_argument("--n_shots", type=int, default=0,
                        help="Number of few-shot examples in prompt")

    # Uncertainty scores to evaluate
    parser.add_argument("--scores", type=str, nargs="+",
                        default=["max_proba_complement", "margin", "doctor", "doctor_normalized"],
                        help="Uncertainty scores to evaluate")

    # Calibration methods
    parser.add_argument("--calibrators", type=str, nargs="+",
                        default=["none", "phc_dp", "phc_ts", "phc_bo", "uniform_mass"],
                        help="Calibration methods to compare")
    parser.add_argument("--n_bins_ece", type=int, default=10,
                        help="Number of bins for ECE computation (Uniform Mass uses Scott's rule)")

    # Output and caching
    parser.add_argument("--output_dir", type=str, default="results/uncertainty_calibration")
    parser.add_argument("--cache_dir", type=str, default="cache/probabilities",
                        help="Directory to cache extracted probabilities")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable probability caching (always run inference)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_cache_path(cache_dir: Path, dataset: str, model: str, split: str,
                   n_samples: int, n_shots: int, seed: int) -> Path:
    """Generate cache file path for probabilities."""
    model_clean = model.replace("/", "_")
    filename = f"{dataset}_{model_clean}_{split}_n{n_samples}_shots{n_shots}_seed{seed}.npz"
    return cache_dir / filename


def load_cached_probabilities(cache_path: Path, logger):
    """Load cached probabilities if available."""
    if cache_path.exists():
        logger.info(f"Loading cached probabilities from {cache_path}")
        data = np.load(cache_path)
        return data["probs"], data["labels"], data["indices"]
    return None, None, None


def save_probabilities_cache(cache_path: Path, probs: np.ndarray,
                             labels: np.ndarray, indices: np.ndarray, logger):
    """Save probabilities to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, probs=probs, labels=labels, indices=indices)
    logger.info(f"Probabilities cached to {cache_path}")


def run_calibration_experiment(
    probs_cal: np.ndarray,
    labels_cal: np.ndarray,
    probs_test: np.ndarray,
    labels_test: np.ndarray,
    score_name: str,
    calibrator_name: str,
    n_bins_ece: int = 10,
    logger=None,
) -> dict:
    """
    Run calibration experiment for a single score and calibrator.

    Fits calibrator on ALL calibration data and evaluates on test data.

    Returns dict with metrics before and after calibration.
    """
    # Compute uncertainty scores
    scores_cal_obj = compute_uncertainty_scores(probs_cal)
    scores_test_obj = compute_uncertainty_scores(probs_test)

    scores_cal = scores_cal_obj.to_dict()[score_name]
    scores_test = scores_test_obj.to_dict()[score_name]

    # Get error indicators
    _, errors_cal = get_predictions_and_errors(probs_cal, labels_cal)
    _, errors_test = get_predictions_and_errors(probs_test, labels_test)

    # Fit calibrator on ALL calibration data
    # Note: Uniform Mass uses Scott's rule for bins (n_bins=None, use_scott_rule=True)
    calibrator = get_calibrator(calibrator_name)
    calibrator.fit(scores_cal, errors_cal)

    # Calibrate test scores
    calibrated_probs = calibrator.calibrate(scores_test)

    # Compute metrics on TEST set
    results = compare_calibration(
        scores=scores_test,
        errors=errors_test,
        calibrated_probs=calibrated_probs,
        n_bins=n_bins_ece,
    )

    # Add calibrator parameters if available
    if hasattr(calibrator, 'get_params'):
        results['calibrator_params'] = calibrator.get_params()

    # Add training info
    if hasattr(calibrator, 'n_iterations_run'):
        results['n_iterations_run'] = calibrator.n_iterations_run
    if hasattr(calibrator, 'n_bins_actual'):
        results['n_bins_actual'] = calibrator.n_bins_actual

    # Add error rate info
    results['error_rate_cal'] = float(errors_cal.mean())
    results['error_rate_test'] = float(errors_test.mean())

    return results


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    logger.info("=" * 60)
    logger.info("Uncertainty Score Calibration Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Affine calibrator: {args.calibrator_checkpoint or 'None (raw probabilities)'}")
    logger.info(f"Calibration samples: {args.n_calibration}")
    logger.info(f"Test samples: {args.n_test}")
    logger.info(f"Shots: {args.n_shots}")
    logger.info(f"Scores: {args.scores}")
    logger.info(f"Calibrators: {args.calibrators}")
    logger.info(f"Cache dir: {cache_dir} (disabled: {args.no_cache})")

    # Load affine calibrator if provided
    affine_calibrator = None
    if args.calibrator_checkpoint:
        logger.info(f"\nLoading affine calibrator from {args.calibrator_checkpoint}...")
        with open(args.calibrator_checkpoint, 'rb') as f:
            ckpt = pickle.load(f)
        affine_calibrator = ckpt['calibrator']
        logger.info(f"  Alpha: {affine_calibrator.alpha:.4f}")
        logger.info(f"  Beta: {affine_calibrator.beta}")

    # Check cache for probabilities
    cal_cache_path = get_cache_path(
        cache_dir, args.dataset, args.model, "calibration",
        args.n_calibration, args.n_shots, args.seed
    )
    test_cache_path = get_cache_path(
        cache_dir, args.dataset, args.model, "test",
        args.n_test, args.n_shots, args.seed
    )

    # Try to load cached probabilities
    cal_probs_raw, cal_labels, cal_indices = None, None, None
    test_probs_raw, test_labels, test_indices = None, None, None

    if not args.no_cache:
        cal_probs_raw, cal_labels, cal_indices = load_cached_probabilities(cal_cache_path, logger)
        test_probs_raw, test_labels, test_indices = load_cached_probabilities(test_cache_path, logger)

    # Determine if we need to load the model
    need_model = cal_probs_raw is None or test_probs_raw is None

    classifier = None
    if need_model:
        logger.info("\nLoading model...")
        classifier = LLMClassifier(model_name=args.model)
    else:
        logger.info("\nUsing cached probabilities - skipping model loading")

    # Load dataset
    # Need enough samples for calibration + test
    total_needed = args.n_calibration + args.n_test + args.n_shots + 100
    dataset = load_dataset_by_name(
        args.dataset,
        n_train=total_needed,
        n_test=args.n_test,
        seed=args.seed,
    )
    logger.info(f"Dataset loaded: {len(dataset.train_texts)} train, {len(dataset.test_texts)} test")

    # Split train data: shots, calibration (use all for fitting)
    rng = np.random.RandomState(args.seed)
    all_indices = rng.permutation(len(dataset.train_texts))

    # Extract shots
    if args.n_shots > 0:
        shot_indices = all_indices[:args.n_shots].tolist()
        remaining_indices = all_indices[args.n_shots:]
    else:
        shot_indices = []
        remaining_indices = all_indices

    # All remaining for calibration (no split - use ALL for fitting)
    cal_indices = remaining_indices[:args.n_calibration].tolist()

    logger.info(f"\nData split:")
    logger.info(f"  Shots: {len(shot_indices)}")
    logger.info(f"  Calibration (ALL for fitting): {len(cal_indices)}")
    logger.info(f"  Test (for evaluation): {len(dataset.test_texts)}")

    # Build preface with shots
    preface = dataset.get_few_shot_preface(
        args.n_shots,
        shot_indices=shot_indices if args.n_shots > 0 else None,
        seed=args.seed
    )

    # Get probabilities for calibration set (from cache or inference)
    if cal_probs_raw is None:
        logger.info("\nExtracting probabilities for calibration set...")
        cal_prompts = [
            dataset.build_prompt(preface, dataset.train_texts[i])
            for i in cal_indices
        ]
        cal_probs_raw = classifier.get_batch_label_probabilities(
            cal_prompts,
            dataset.label_names,
            show_progress=True,
        )
        cal_labels = np.array([dataset.train_labels[i] for i in cal_indices])

        # Cache the probabilities
        if not args.no_cache:
            save_probabilities_cache(
                cal_cache_path, cal_probs_raw, cal_labels,
                np.array(cal_indices), logger
            )
    else:
        logger.info("\nUsing cached calibration probabilities")

    # Apply affine calibration if available
    if affine_calibrator is not None:
        logger.info("Applying affine calibration to calibration probabilities...")
        cal_probs = affine_calibrator.calibrate(cal_probs_raw)
    else:
        cal_probs = cal_probs_raw

    # Get probabilities for test set (from cache or inference)
    if test_probs_raw is None:
        logger.info("\nExtracting probabilities for test set...")
        test_prompts = dataset.build_prompts_for_split(preface, split="test")
        test_probs_raw = classifier.get_batch_label_probabilities(
            test_prompts,
            dataset.label_names,
            show_progress=True,
        )
        test_labels = np.array(dataset.test_labels)

        # Cache the probabilities
        if not args.no_cache:
            save_probabilities_cache(
                test_cache_path, test_probs_raw, test_labels,
                np.arange(len(test_labels)), logger
            )
    else:
        logger.info("\nUsing cached test probabilities")

    # Apply affine calibration if available
    if affine_calibrator is not None:
        logger.info("Applying affine calibration to test probabilities...")
        test_probs = affine_calibrator.calibrate(test_probs_raw)
    else:
        test_probs = test_probs_raw

    # Store all results
    all_results = {
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "calibrator_checkpoint": args.calibrator_checkpoint,
            "n_calibration": args.n_calibration,
            "n_test": args.n_test,
            "n_shots": args.n_shots,
            "n_bins_ece": args.n_bins_ece,
            "seed": args.seed,
        },
        "results": {},
    }

    # Run experiments for each score and calibrator
    logger.info("\n" + "=" * 60)
    logger.info("Running calibration experiments...")
    logger.info("=" * 60)
    logger.info("Note: Calibrators are fitted on ALL calibration data, evaluated on test set")
    logger.info("      Uniform Mass uses Scott's rule for bins: B = 2 * N^(1/3)")

    for score_name in args.scores:
        logger.info(f"\n--- Score: {score_name} ---")
        all_results["results"][score_name] = {}

        for calibrator_name in args.calibrators:
            logger.info(f"  Calibrator: {calibrator_name}")

            results = run_calibration_experiment(
                probs_cal=cal_probs,
                labels_cal=cal_labels,
                probs_test=test_probs,
                labels_test=test_labels,
                score_name=score_name,
                calibrator_name=calibrator_name,
                n_bins_ece=args.n_bins_ece,
                logger=logger,
            )

            all_results["results"][score_name][calibrator_name] = results

            # Print summary
            before = results["before"]
            after = results["after"]
            logger.info(f"    ROCAUC: {before['rocauc']:.4f}")
            logger.info(f"    ECE:    {before['ece']:.4f} -> {after['ece']:.4f}")
            logger.info(f"    BCE:    {before['binary_cross_entropy']:.4f} -> {after['binary_cross_entropy']:.4f}")

    # Save results
    results_file = output_dir / f"{args.dataset}_uncertainty_calibration.json"
    save_results(all_results, results_file)
    logger.info(f"\nResults saved to {results_file}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: ECE (before -> after)")
    logger.info("=" * 70)

    header = f"{'Score':<25}"
    for cal in args.calibrators:
        header += f" {cal:<15}"
    logger.info(header)
    logger.info("-" * 70)

    for score_name in args.scores:
        row = f"{score_name:<25}"
        for cal in args.calibrators:
            res = all_results["results"][score_name][cal]
            before_ece = res["before"]["ece"]
            after_ece = res["after"]["ece"]
            row += f" {before_ece:.3f}->{after_ece:.3f} "
        logger.info(row)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Binary Cross-Entropy (before -> after)")
    logger.info("=" * 70)

    logger.info(header)
    logger.info("-" * 70)

    for score_name in args.scores:
        row = f"{score_name:<25}"
        for cal in args.calibrators:
            res = all_results["results"][score_name][cal]
            before_bce = res["before"]["binary_cross_entropy"]
            after_bce = res["after"]["binary_cross_entropy"]
            row += f" {before_bce:.3f}->{after_bce:.3f} "
        logger.info(row)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: ROCAUC (unchanged by calibration)")
    logger.info("=" * 70)

    for score_name in args.scores:
        res = all_results["results"][score_name][args.calibrators[0]]
        rocauc = res["before"]["rocauc"]
        error_rate = res["error_rate_test"]
        logger.info(f"{score_name:<25}: ROCAUC={rocauc:.4f}, Error Rate={error_rate:.4f}")


if __name__ == "__main__":
    main()
