#!/usr/bin/env python3
"""
Step 1: Finetune LLM posteriors with Affine Calibration.

This script:
1. Loads the base LLM (GPT-2 XL)
2. Extracts raw probabilities on training data
3. Fits AffineCalibrator (learns α and β parameters)
4. Saves calibration parameters to a checkpoint file

The checkpoint can then be loaded by run_uncertainty_calibration.py
to get calibrated probabilities for uncertainty score experiments.
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
from src.evaluation import compute_metrics
from src.utils import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune LLM with Affine Calibration")

    # Model and data
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--model", type=str, default="gpt2-xl")

    # Data sizes
    parser.add_argument("--n_train", type=int, default=600,
                        help="Number of samples for training affine calibration")
    parser.add_argument("--n_val", type=int, default=200,
                        help="Number of samples for validation")

    # Few-shot settings
    parser.add_argument("--n_shots", type=int, default=0,
                        help="Number of few-shot examples in prompt")

    # Affine calibration settings
    parser.add_argument("--learn_alpha", action="store_true", default=True,
                        help="Learn alpha parameter (temperature)")
    parser.add_argument("--n_iterations", type=int, default=1000,
                        help="Maximum number of optimization iterations")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate for optimization")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (stop if loss doesn't decrease)")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 1: Finetune LLM with Affine Calibration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Training samples: {args.n_train}")
    logger.info(f"Validation samples: {args.n_val}")
    logger.info(f"Shots: {args.n_shots}")
    logger.info(f"Learn alpha: {args.learn_alpha}")
    logger.info(f"Seed: {args.seed}")

    # Load model
    logger.info("\nLoading model...")
    classifier = LLMClassifier(model_name=args.model)

    # Load dataset
    total_needed = args.n_train + args.n_val + args.n_shots + 100
    dataset = load_dataset_by_name(
        args.dataset,
        n_train=total_needed,
        n_test=args.n_val,  # Use test split for validation
        seed=args.seed,
    )
    logger.info(f"Dataset loaded: {len(dataset.train_texts)} train, {len(dataset.test_texts)} test")

    # Split data
    rng = np.random.RandomState(args.seed)
    all_indices = rng.permutation(len(dataset.train_texts))

    if args.n_shots > 0:
        shot_indices = all_indices[:args.n_shots].tolist()
        train_indices = all_indices[args.n_shots:args.n_shots + args.n_train].tolist()
    else:
        shot_indices = []
        train_indices = all_indices[:args.n_train].tolist()

    logger.info(f"\nData split:")
    logger.info(f"  Shots: {len(shot_indices)}")
    logger.info(f"  Train (for affine cal): {len(train_indices)}")
    logger.info(f"  Validation: {len(dataset.test_texts)}")

    # Build preface with shots
    preface = dataset.get_few_shot_preface(
        args.n_shots,
        shot_indices=shot_indices if args.n_shots > 0 else None,
        seed=args.seed
    )

    # Get raw probabilities on training data
    logger.info("\nExtracting raw probabilities on training data...")
    train_prompts = [
        dataset.build_prompt(preface, dataset.train_texts[i])
        for i in train_indices
    ]
    train_probs_raw = classifier.get_batch_label_probabilities(
        train_prompts,
        dataset.label_names,
        show_progress=True,
    )
    train_labels = np.array([dataset.train_labels[i] for i in train_indices])

    # Evaluate raw probabilities
    logger.info("\nRaw model performance on training data:")
    raw_metrics = compute_metrics(train_probs_raw, train_labels)
    logger.info(f"  Accuracy: {raw_metrics.accuracy:.4f}")
    logger.info(f"  Error Rate: {raw_metrics.error_rate:.4f}")
    logger.info(f"  ECE: {raw_metrics.ece:.4f}")
    logger.info(f"  Cross-Entropy: {raw_metrics.cross_entropy:.4f}")

    # Fit Affine Calibrator
    logger.info("\nFitting Affine Calibrator...")
    logger.info(f"  Max iterations: {args.n_iterations}")
    logger.info(f"  Early stopping patience: {args.patience}")
    calibrator = AffineCalibrator(
        learn_alpha=args.learn_alpha,
        n_iterations=args.n_iterations,
        lr=args.lr,
        patience=args.patience,
    )
    calibrator.fit(train_probs_raw, train_labels)

    logger.info(f"  Iterations run: {calibrator.n_iterations_run}")
    logger.info(f"  Learned alpha: {calibrator.alpha:.4f}")
    logger.info(f"  Learned beta: {calibrator.beta}")

    # Report loss dynamics
    logger.info("\nLoss dynamics during training:")
    loss_history = calibrator.loss_history
    logger.info(f"  Initial loss: {loss_history[0]:.4f}")
    logger.info(f"  Final loss: {loss_history[-1]:.4f}")
    logger.info(f"  Best loss: {min(loss_history):.4f}")

    # Apply calibration to training data
    train_probs_cal = calibrator.calibrate(train_probs_raw)
    cal_metrics = compute_metrics(train_probs_cal, train_labels)
    logger.info("\nCalibrated performance on training data:")
    logger.info(f"  Accuracy: {cal_metrics.accuracy:.4f}")
    logger.info(f"  Error Rate: {cal_metrics.error_rate:.4f}")
    logger.info(f"  ECE: {cal_metrics.ece:.4f}")
    logger.info(f"  Cross-Entropy: {cal_metrics.cross_entropy:.4f}")

    # Validate on held-out data
    logger.info("\nExtracting probabilities on validation data...")
    val_prompts = dataset.build_prompts_for_split(preface, split="test")
    val_probs_raw = classifier.get_batch_label_probabilities(
        val_prompts,
        dataset.label_names,
        show_progress=True,
    )
    val_labels = np.array(dataset.test_labels)

    # Raw performance on validation
    val_raw_metrics = compute_metrics(val_probs_raw, val_labels)
    logger.info("\nRaw model performance on validation:")
    logger.info(f"  Accuracy: {val_raw_metrics.accuracy:.4f}")
    logger.info(f"  Error Rate: {val_raw_metrics.error_rate:.4f}")
    logger.info(f"  ECE: {val_raw_metrics.ece:.4f}")
    logger.info(f"  Cross-Entropy: {val_raw_metrics.cross_entropy:.4f}")

    # Calibrated performance on validation
    val_probs_cal = calibrator.calibrate(val_probs_raw)
    val_cal_metrics = compute_metrics(val_probs_cal, val_labels)
    logger.info("\nCalibrated performance on validation:")
    logger.info(f"  Accuracy: {val_cal_metrics.accuracy:.4f}")
    logger.info(f"  Error Rate: {val_cal_metrics.error_rate:.4f}")
    logger.info(f"  ECE: {val_cal_metrics.ece:.4f}")
    logger.info(f"  Cross-Entropy: {val_cal_metrics.cross_entropy:.4f}")

    # Save checkpoint
    checkpoint = {
        "model_name": args.model,
        "dataset": args.dataset,
        "n_train": args.n_train,
        "n_shots": args.n_shots,
        "seed": args.seed,
        "preface": preface,
        "calibrator": {
            "type": "AffineCalibrator",
            "alpha": float(calibrator.alpha),
            "beta": calibrator.beta.tolist(),
            "learn_alpha": args.learn_alpha,
            "n_iterations_run": calibrator.n_iterations_run,
        },
        "loss_history": [float(l) for l in calibrator.loss_history],
        "metrics": {
            "train_raw": raw_metrics.to_dict(),
            "train_calibrated": cal_metrics.to_dict(),
            "val_raw": val_raw_metrics.to_dict(),
            "val_calibrated": val_cal_metrics.to_dict(),
        },
    }

    # Save as JSON (human readable)
    checkpoint_name = f"{args.dataset}_affine_seed{args.seed}"
    json_path = output_dir / f"{checkpoint_name}.json"
    with open(json_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"\nCheckpoint (JSON) saved to: {json_path}")

    # Save as pickle (for loading calibrator object)
    pkl_path = output_dir / f"{checkpoint_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            "calibrator": calibrator,
            "preface": preface,
            "config": checkpoint,
        }, f)
    logger.info(f"Checkpoint (pickle) saved to: {pkl_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY (Validation Set)")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'Raw':>12} {'Calibrated':>12} {'Improvement':>12}")
    logger.info("-" * 60)

    for metric in ['accuracy', 'error_rate', 'ece', 'cross_entropy']:
        raw_val = val_raw_metrics.to_dict()[metric]
        cal_val = val_cal_metrics.to_dict()[metric]
        if metric == 'accuracy':
            imp = cal_val - raw_val
            imp_str = f"+{imp:.4f}" if imp > 0 else f"{imp:.4f}"
        else:
            imp = raw_val - cal_val
            imp_str = f"-{abs(imp):.4f}" if imp > 0 else f"+{abs(imp):.4f}"
        logger.info(f"{metric:<20} {raw_val:>12.4f} {cal_val:>12.4f} {imp_str:>12}")

    # Print loss dynamics summary
    logger.info("\n" + "=" * 60)
    logger.info("LOSS DYNAMICS")
    logger.info("=" * 60)
    logger.info(f"Iterations run: {calibrator.n_iterations_run}")
    logger.info(f"Initial loss: {loss_history[0]:.4f}")
    logger.info(f"Final loss: {loss_history[-1]:.4f}")
    logger.info(f"Best loss: {min(loss_history):.4f}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Checkpoint saved. Use with run_uncertainty_calibration.py:")
    logger.info(f"  --calibrator_checkpoint {pkl_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
