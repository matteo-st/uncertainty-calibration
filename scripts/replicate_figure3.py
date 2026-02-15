#!/usr/bin/env python3
"""
Replicate Figure 3 from Estienne et al. (2023) for AGNews dataset.

Figure 3 shows Cross-Entropy and Error Rate vs. number of shots (0, 1, 4, 8)
with 600 training samples.

Methods: No Adaptation, UCPA, SUCPA, Calibration (affine with α and β)
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
from src.calibration import (
    NoCalibration,
    UCPACalibrator,
    SUCPACalibrator,
    AffineCalibrator,
)
from src.evaluation import compute_metrics
from src.utils import save_results, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Replicate Figure 3")
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--model", type=str, default="gpt2-xl")
    parser.add_argument("--n_train", type=int, default=600)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_shots_list", type=int, nargs="+", default=[0, 1, 4, 8])
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/figure3")
    parser.add_argument("--base_seed", type=int, default=42)
    return parser.parse_args()


def run_for_n_shots(classifier, dataset, n_shots, n_train, seed, logger):
    """Run inference and calibration for a specific n_shots configuration."""

    # Determine which samples to use for shots vs calibration
    if n_shots > 0:
        shot_indices = list(range(n_shots))
        # Use samples after the shots for calibration
        train_indices = list(range(n_shots, min(n_shots + n_train, len(dataset.train_texts))))
    else:
        shot_indices = []
        train_indices = list(range(min(n_train, len(dataset.train_texts))))

    # Build preface with shots
    preface = dataset.get_few_shot_preface(n_shots, shot_indices=shot_indices, seed=seed)

    # Get train probabilities (for calibration)
    train_prompts = [
        dataset.build_prompt(preface, dataset.train_texts[i])
        for i in train_indices
    ]
    train_probs = classifier.get_batch_label_probabilities(
        train_prompts,
        dataset.label_names,
        show_progress=False,
    )
    train_labels = np.array([dataset.train_labels[i] for i in train_indices])

    # Get test probabilities
    test_prompts = dataset.build_prompts_for_split(preface, split="test")
    test_probs = classifier.get_batch_label_probabilities(
        test_prompts,
        dataset.label_names,
        show_progress=False,
    )
    test_labels = np.array(dataset.test_labels)

    # Define calibrators (matching paper's Figure 3)
    calibrators = {
        "no_adaptation": NoCalibration(),
        "ucpa": UCPACalibrator(),
        "sucpa": SUCPACalibrator(),
        "calibration": AffineCalibrator(learn_alpha=True),  # Full affine with α and β
    }

    results = {}
    for name, calibrator in calibrators.items():
        calibrator.fit(train_probs, train_labels)
        calibrated_probs = calibrator.calibrate(test_probs)
        metrics = compute_metrics(calibrated_probs, test_labels)
        results[name] = metrics.to_dict()

    return results


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Replicating Figure 3: AGNews - Error Rate & CE vs. Shots")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Training samples: {args.n_train}")
    logger.info(f"Test samples: {args.n_test}")
    logger.info(f"Shots: {args.n_shots_list}")
    logger.info(f"Seeds: {args.n_seeds}")

    # Load model
    logger.info("\nLoading model...")
    classifier = LLMClassifier(model_name=args.model)

    # Load dataset with enough samples
    max_train = args.n_train + max(args.n_shots_list)
    dataset = load_dataset_by_name(
        args.dataset,
        n_train=max_train,
        n_test=args.n_test,
        seed=args.base_seed,
    )
    logger.info(f"Dataset loaded: {len(dataset.train_texts)} train, {len(dataset.test_texts)} test")

    # Results storage
    all_results = {
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_shots_list": args.n_shots_list,
            "n_seeds": args.n_seeds,
        },
        "results_by_shots": {},
    }

    # Run for each n_shots
    for n_shots in args.n_shots_list:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running {n_shots}-shot experiments...")
        logger.info(f"{'='*40}")

        seed_results = []

        for seed_idx in tqdm(range(args.n_seeds), desc=f"{n_shots}-shot"):
            seed = args.base_seed + seed_idx
            set_seed(seed)

            results = run_for_n_shots(
                classifier, dataset, n_shots, args.n_train, seed, logger
            )
            seed_results.append(results)

        # Aggregate results across seeds
        aggregated = {}
        methods = seed_results[0].keys()

        for method in methods:
            aggregated[method] = {}
            for metric in ["accuracy", "error_rate", "cross_entropy", "normalized_cross_entropy"]:
                values = [r[method][metric] for r in seed_results]
                aggregated[method][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        all_results["results_by_shots"][str(n_shots)] = aggregated

        # Print summary for this n_shots
        logger.info(f"\n{n_shots}-shot Results:")
        logger.info(f"{'Method':<20} {'Error Rate':>15} {'Cross-Entropy':>15}")
        logger.info("-" * 55)
        for method in methods:
            er = aggregated[method]["error_rate"]
            ce = aggregated[method]["cross_entropy"]
            logger.info(f"{method:<20} {er['mean']:>7.4f} ± {er['std']:.4f} {ce['mean']:>7.4f} ± {ce['std']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}_figure3_results.json"
    save_results(all_results, output_file)
    logger.info(f"\nResults saved to {output_file}")

    # Print final summary table
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS - Figure 3 Replication")
    logger.info("=" * 70)

    logger.info("\nError Rate:")
    logger.info(f"{'Shots':<8} {'No Adapt':>12} {'UCPA':>12} {'SUCPA':>12} {'Calibration':>12}")
    logger.info("-" * 60)
    for n_shots in args.n_shots_list:
        row = f"{n_shots:<8}"
        for method in ["no_adaptation", "ucpa", "sucpa", "calibration"]:
            er = all_results["results_by_shots"][str(n_shots)][method]["error_rate"]
            row += f" {er['mean']:>5.3f}±{er['std']:.3f}"
        logger.info(row)

    logger.info("\nCross-Entropy:")
    logger.info(f"{'Shots':<8} {'No Adapt':>12} {'UCPA':>12} {'SUCPA':>12} {'Calibration':>12}")
    logger.info("-" * 60)
    for n_shots in args.n_shots_list:
        row = f"{n_shots:<8}"
        for method in ["no_adaptation", "ucpa", "sucpa", "calibration"]:
            ce = all_results["results_by_shots"][str(n_shots)][method]["cross_entropy"]
            row += f" {ce['mean']:>5.3f}±{ce['std']:.3f}"
        logger.info(row)


if __name__ == "__main__":
    main()
