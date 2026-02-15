#!/usr/bin/env python3
"""
Run full experiment: inference + calibration for multiple configurations.

This script runs the complete pipeline for generating results similar to
Figures 2 and 3 in the paper.

Usage:
    python scripts/run_experiment.py --dataset agnews --model gpt2-xl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.models import LLMClassifier
from src.data import load_dataset_by_name
from src.calibration import (
    NoCalibration,
    ContentFreeCalibrator,
    UCPANaiveCalibrator,
    UCPACalibrator,
    SUCPANaiveCalibrator,
    SUCPACalibrator,
)
from src.evaluation import compute_metrics
from src.utils import save_results, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run full calibration experiment")
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--model", type=str, default="gpt2-xl")
    parser.add_argument("--n_shots_list", type=int, nargs="+", default=[0, 1, 4, 8])
    parser.add_argument("--n_train_list", type=int, nargs="+",
                        default=[10, 20, 40, 80, 200, 400, 600])
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random seeds for train subsampling")
    parser.add_argument("--output_dir", type=str, default="results/experiments")
    parser.add_argument("--base_seed", type=int, default=42)
    return parser.parse_args()


def run_single_config(
    classifier,
    dataset,
    n_shots: int,
    n_train: int,
    seed: int,
    logger,
):
    """Run calibration for a single configuration."""

    # Build preface
    if n_shots > 0:
        shot_indices = list(range(n_shots))
        train_indices = list(range(n_shots, min(n_shots + n_train, len(dataset.train_texts))))
    else:
        shot_indices = []
        train_indices = list(range(min(n_train, len(dataset.train_texts))))

    preface = dataset.get_few_shot_preface(n_shots, shot_indices=shot_indices, seed=seed)

    # Get train probabilities
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

    # Get content-free probabilities
    content_free_inputs = ["[MASK]", "N/A", ""]
    content_free_prompts = [
        dataset.build_prompt(preface, cf_input)
        for cf_input in content_free_inputs
    ]
    content_free_probs = classifier.get_batch_label_probabilities(
        content_free_prompts,
        dataset.label_names,
        show_progress=False,
    )

    # Run calibration methods
    calibrators = {
        "no_calibration": NoCalibration(),
        "content_free": ContentFreeCalibrator(content_free_probs),
        "ucpa_naive": UCPANaiveCalibrator(),
        "ucpa": UCPACalibrator(),
        "sucpa_naive": SUCPANaiveCalibrator(),
        "sucpa": SUCPACalibrator(),
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

    logger.info(f"Running experiment: {args.dataset} with {args.model}")
    logger.info(f"Shots: {args.n_shots_list}")
    logger.info(f"Train sizes: {args.n_train_list}")
    logger.info(f"Seeds: {args.n_seeds}")

    # Load model once
    classifier = LLMClassifier(model_name=args.model)

    # Load dataset with max samples
    max_train = max(args.n_train_list) + max(args.n_shots_list)
    dataset = load_dataset_by_name(
        args.dataset,
        n_train=max_train,
        n_test=args.n_test,
        seed=args.base_seed,
    )

    all_results = {
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "n_shots_list": args.n_shots_list,
            "n_train_list": args.n_train_list,
            "n_test": args.n_test,
            "n_seeds": args.n_seeds,
        },
        "results": {},
    }

    # Run experiments
    for n_shots in args.n_shots_list:
        logger.info(f"\n=== {n_shots}-shot ===")
        all_results["results"][f"{n_shots}shot"] = {}

        for n_train in args.n_train_list:
            logger.info(f"  n_train={n_train}")

            seed_results = []
            for seed_idx in range(args.n_seeds):
                seed = args.base_seed + seed_idx
                set_seed(seed)

                results = run_single_config(
                    classifier, dataset, n_shots, n_train, seed, logger
                )
                seed_results.append(results)

            # Aggregate across seeds
            aggregated = {}
            for method in seed_results[0].keys():
                aggregated[method] = {}
                for metric in seed_results[0][method].keys():
                    values = [r[method][metric] for r in seed_results]
                    aggregated[method][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }

            all_results["results"][f"{n_shots}shot"][f"n_train_{n_train}"] = aggregated

            # Print summary
            baseline = aggregated["no_calibration"]["error_rate"]["mean"]
            ucpa = aggregated["ucpa"]["error_rate"]["mean"]
            logger.info(f"    Baseline error: {baseline:.4f}, UCPA error: {ucpa:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}_{args.model.replace('/', '_')}_experiment.json"
    save_results(all_results, output_file)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
