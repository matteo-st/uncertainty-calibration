#!/usr/bin/env python3
"""
Apply calibration methods to saved probabilities and evaluate.

Usage:
    python scripts/run_calibration.py --probs_dir results/probabilities --output_dir results/calibration
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.calibration import (
    NoCalibration,
    ContentFreeCalibrator,
    UCPANaiveCalibrator,
    UCPACalibrator,
    SUCPANaiveCalibrator,
    SUCPACalibrator,
    AffineCalibrator,
)
from src.evaluation import compute_metrics, bootstrap_metrics
from src.utils import load_probabilities, save_results, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibration experiments")
    parser.add_argument("--probs_dir", type=str, required=True,
                        help="Directory with saved probabilities")
    parser.add_argument("--output_dir", type=str, default="results/calibration")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Prefix for probability files (e.g., 'agnews_gpt2-xl_0shot_seed42')")
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    probs_dir = Path(args.probs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find probability files
    if args.prefix:
        train_file = probs_dir / f"{args.prefix}_train.npz"
        test_file = probs_dir / f"{args.prefix}_test.npz"
        cf_file = probs_dir / f"{args.prefix}_contentfree.npz"
    else:
        # Find first matching set
        train_files = list(probs_dir.glob("*_train.npz"))
        if not train_files:
            raise FileNotFoundError(f"No probability files found in {probs_dir}")
        prefix = train_files[0].stem.replace("_train", "")
        train_file = train_files[0]
        test_file = probs_dir / f"{prefix}_test.npz"
        cf_file = probs_dir / f"{prefix}_contentfree.npz"

    logger.info(f"Loading probabilities from {probs_dir}")
    train_data = load_probabilities(train_file)
    test_data = load_probabilities(test_file)
    cf_data = np.load(cf_file)

    train_probs = train_data["probs"]
    train_labels = train_data["labels"]
    test_probs = test_data["probs"]
    test_labels = test_data["labels"]
    content_free_probs = cf_data["probs"]

    logger.info(f"Train: {train_probs.shape}, Test: {test_probs.shape}")

    # Define calibrators
    calibrators = {
        "no_calibration": NoCalibration(),
        "content_free_idk": ContentFreeCalibrator(content_free_probs[0]),  # [MASK]
        "content_free_all": ContentFreeCalibrator(content_free_probs),  # Average of all
        "ucpa_naive": UCPANaiveCalibrator(),
        "ucpa": UCPACalibrator(),
        "sucpa_naive": SUCPANaiveCalibrator(),
        "sucpa": SUCPACalibrator(),
        "affine_alpha1": AffineCalibrator(learn_alpha=False),
        "affine_full": AffineCalibrator(learn_alpha=True),
    }

    results = {}

    for name, calibrator in calibrators.items():
        logger.info(f"\n=== {name} ===")

        # Fit calibrator
        if name.startswith("content_free"):
            calibrator.fit(train_probs)  # Doesn't use train data
        elif name in ["affine_alpha1", "affine_full"]:
            calibrator.fit(train_probs, train_labels)  # Supervised
        else:
            # UCPA variants: unsupervised but uses train probs
            # SUCPA variants: uses train labels for prior estimation
            calibrator.fit(train_probs, train_labels)

        # Calibrate test probabilities
        calibrated_probs = calibrator.calibrate(test_probs)

        # Compute metrics
        metrics = compute_metrics(calibrated_probs, test_labels)
        bootstrap = bootstrap_metrics(calibrated_probs, test_labels, args.n_bootstrap, args.seed)

        results[name] = {
            "metrics": metrics.to_dict(),
            "bootstrap": bootstrap,
        }

        logger.info(f"  Accuracy: {metrics.accuracy:.4f} (+/- {bootstrap['accuracy']['std']:.4f})")
        logger.info(f"  Error Rate: {metrics.error_rate:.4f}")
        logger.info(f"  Cross-Entropy: {metrics.cross_entropy:.4f}")
        logger.info(f"  Normalized CE: {metrics.normalized_cross_entropy:.4f}")

    # Save results
    output_file = output_dir / f"{prefix}_calibration_results.json"
    save_results(results, output_file)
    logger.info(f"\nResults saved to {output_file}")

    # Print comparison table
    logger.info("\n=== Comparison ===")
    logger.info(f"{'Method':<20} {'Accuracy':>10} {'Error Rate':>12} {'Cross-Entropy':>14} {'Norm CE':>10}")
    logger.info("-" * 70)
    for name, res in results.items():
        m = res["metrics"]
        logger.info(f"{name:<20} {m['accuracy']:>10.4f} {m['error_rate']:>12.4f} {m['cross_entropy']:>14.4f} {m['normalized_cross_entropy']:>10.4f}")


if __name__ == "__main__":
    main()
