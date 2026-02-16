#!/usr/bin/env python3
"""
Run uncertainty score calibration experiments for error prediction.

This script:
1. Loads a model (base or finetuned checkpoint)
2. Extracts class probabilities on calibration samples
3. Computes uncertainty scores (1-max_proba, margin, doctor)
4. Splits data for calibrator training/testing
5. Fits calibrators (PHC-DP, Uniform Mass, etc.)
6. Evaluates calibration quality (ROCAUC, ECE, Cross-Entropy)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from tqdm import tqdm

from src.models import LLMClassifier
from src.data import load_dataset_by_name
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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to finetuned model checkpoint (optional)")

    # Data sizes
    parser.add_argument("--n_calibration", type=int, default=1000,
                        help="Number of samples for calibration")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Number of samples for final testing")
    parser.add_argument("--calibration_split", type=float, default=0.7,
                        help="Fraction of calibration data for training calibrator")

    # Few-shot settings
    parser.add_argument("--n_shots", type=int, default=0,
                        help="Number of few-shot examples in prompt")

    # Uncertainty scores to evaluate
    parser.add_argument("--scores", type=str, nargs="+",
                        default=["max_proba_complement", "margin", "doctor", "doctor_normalized"],
                        help="Uncertainty scores to evaluate")

    # Calibration methods
    parser.add_argument("--calibrators", type=str, nargs="+",
                        default=["none", "phc_dp", "phc_ts", "uniform_mass"],
                        help="Calibration methods to compare")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for uniform mass and ECE")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/uncertainty_calibration")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_calibration_experiment(
    probs_train: np.ndarray,
    labels_train: np.ndarray,
    probs_test: np.ndarray,
    labels_test: np.ndarray,
    score_name: str,
    calibrator_name: str,
    n_bins: int = 10,
    logger=None,
) -> dict:
    """
    Run calibration experiment for a single score and calibrator.

    Returns dict with metrics before and after calibration.
    """
    # Compute uncertainty scores
    scores_train_obj = compute_uncertainty_scores(probs_train)
    scores_test_obj = compute_uncertainty_scores(probs_test)

    scores_train = scores_train_obj.to_dict()[score_name]
    scores_test = scores_test_obj.to_dict()[score_name]

    # Get error indicators
    _, errors_train = get_predictions_and_errors(probs_train, labels_train)
    _, errors_test = get_predictions_and_errors(probs_test, labels_test)

    # Fit calibrator
    calibrator = get_calibrator(calibrator_name, n_bins=n_bins)
    calibrator.fit(scores_train, errors_train)

    # Calibrate test scores
    calibrated_probs = calibrator.calibrate(scores_test)

    # Compute metrics
    results = compare_calibration(
        scores=scores_test,
        errors=errors_test,
        calibrated_probs=calibrated_probs,
        n_bins=n_bins,
    )

    # Add calibrator parameters if available
    if hasattr(calibrator, 'get_params'):
        results['calibrator_params'] = calibrator.get_params()

    # Add error rate info
    results['error_rate_train'] = float(errors_train.mean())
    results['error_rate_test'] = float(errors_test.mean())

    return results


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Uncertainty Score Calibration Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint or 'None (base model)'}")
    logger.info(f"Calibration samples: {args.n_calibration}")
    logger.info(f"Test samples: {args.n_test}")
    logger.info(f"Shots: {args.n_shots}")
    logger.info(f"Scores: {args.scores}")
    logger.info(f"Calibrators: {args.calibrators}")

    # Load model
    logger.info("\nLoading model...")
    classifier = LLMClassifier(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )

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

    # Split train data: shots, calibration_train, calibration_test
    rng = np.random.RandomState(args.seed)
    all_indices = rng.permutation(len(dataset.train_texts))

    # Extract shots
    if args.n_shots > 0:
        shot_indices = all_indices[:args.n_shots].tolist()
        remaining_indices = all_indices[args.n_shots:]
    else:
        shot_indices = []
        remaining_indices = all_indices

    # Split remaining into calibration train/test
    n_cal_train = int(args.n_calibration * args.calibration_split)
    n_cal_test = args.n_calibration - n_cal_train

    cal_train_indices = remaining_indices[:n_cal_train].tolist()
    cal_test_indices = remaining_indices[n_cal_train:n_cal_train + n_cal_test].tolist()

    logger.info(f"\nData split:")
    logger.info(f"  Shots: {len(shot_indices)}")
    logger.info(f"  Calibration train: {len(cal_train_indices)}")
    logger.info(f"  Calibration test: {len(cal_test_indices)}")
    logger.info(f"  Final test: {len(dataset.test_texts)}")

    # Build preface with shots
    preface = dataset.get_few_shot_preface(
        args.n_shots,
        shot_indices=shot_indices if args.n_shots > 0 else None,
        seed=args.seed
    )

    # Get probabilities for calibration train
    logger.info("\nExtracting probabilities for calibration train...")
    cal_train_prompts = [
        dataset.build_prompt(preface, dataset.train_texts[i])
        for i in cal_train_indices
    ]
    cal_train_probs = classifier.get_batch_label_probabilities(
        cal_train_prompts,
        dataset.label_names,
        show_progress=True,
    )
    cal_train_labels = np.array([dataset.train_labels[i] for i in cal_train_indices])

    # Get probabilities for calibration test
    logger.info("\nExtracting probabilities for calibration test...")
    cal_test_prompts = [
        dataset.build_prompt(preface, dataset.train_texts[i])
        for i in cal_test_indices
    ]
    cal_test_probs = classifier.get_batch_label_probabilities(
        cal_test_prompts,
        dataset.label_names,
        show_progress=True,
    )
    cal_test_labels = np.array([dataset.train_labels[i] for i in cal_test_indices])

    # Store all results
    all_results = {
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "checkpoint": args.checkpoint,
            "n_calibration": args.n_calibration,
            "n_test": args.n_test,
            "calibration_split": args.calibration_split,
            "n_shots": args.n_shots,
            "n_bins": args.n_bins,
            "seed": args.seed,
        },
        "results": {},
    }

    # Run experiments for each score and calibrator
    logger.info("\n" + "=" * 60)
    logger.info("Running calibration experiments...")
    logger.info("=" * 60)

    for score_name in args.scores:
        logger.info(f"\n--- Score: {score_name} ---")
        all_results["results"][score_name] = {}

        for calibrator_name in args.calibrators:
            logger.info(f"  Calibrator: {calibrator_name}")

            results = run_calibration_experiment(
                probs_train=cal_train_probs,
                labels_train=cal_train_labels,
                probs_test=cal_test_probs,
                labels_test=cal_test_labels,
                score_name=score_name,
                calibrator_name=calibrator_name,
                n_bins=args.n_bins,
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
