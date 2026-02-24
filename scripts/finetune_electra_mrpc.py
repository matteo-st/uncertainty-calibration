#!/usr/bin/env python3
"""
Fine-tune ELECTRA on MRPC and extract features for Mahalanobis distance.

This script implements Step 5 of the ELECTRA/MRPC experiment plan:
1. Load MRPC data with train/cal/test splits
2. Fine-tune ELECTRA on the training split
3. Evaluate classification accuracy on all splits
4. Extract penultimate-layer features from all splits
5. Fit MahalanobisScorer on training features
6. Compute MD scores on calibration and test sets
7. Save everything to cache (model, features, scorer, scores, probabilities)

Supports two model variants:
- MD (no SN): Standard ELECTRA, lr=5e-5, epochs=12
- MD SN: ELECTRA with spectral normalization, lr=3e-5, epochs=11

Usage:
    # Standard MD variant
    python finetune_electra_mrpc.py --config configs/mrpc_electra.yaml

    # MD SN variant
    python finetune_electra_mrpc.py --config configs/mrpc_electra.yaml \
        --use_spectral_norm --learning_rate 3e-5 --num_train_epochs 11

    # Quick test with small data
    python finetune_electra_mrpc.py --n_cal 50 --num_train_epochs 1

Prerequisite: None (this is the first step of the encoder experiment).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import logging

from src.encoder_data import load_mrpc
from src.encoder_models import EncoderClassifier
from src.mahalanobis import MahalanobisScorer
from src.uncertainty import compute_uncertainty_scores, get_predictions_and_errors
from src.utils import set_seed, setup_logging, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ELECTRA on MRPC and extract MD features"
    )

    # Configuration file (optional, CLI args override config)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Model
    parser.add_argument("--model_name", type=str,
                        default="google/electra-base-discriminator")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--use_spectral_norm", action="store_true",
                        help="Apply spectral normalization (MD SN variant)")

    # Data
    parser.add_argument("--n_cal", type=int, default=600,
                        help="Number of calibration samples from MRPC train")
    parser.add_argument("--max_length", type=int, default=128)

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=12)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Output
    parser.add_argument("--cache_dir", type=str, default="cache/encoder",
                        help="Directory to save cached outputs")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/encoder",
                        help="Directory for training checkpoints")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, load existing model checkpoint")

    return parser.parse_args()


def apply_config(args, config: dict) -> None:
    """
    Apply YAML config values to args, without overriding CLI-provided values.

    Config structure matches configs/mrpc_electra.yaml.
    """
    # Dataset config
    ds = config.get("dataset", {})
    if args.n_cal == 600 and "n_cal" in ds:
        args.n_cal = ds["n_cal"]
    if args.max_length == 128 and "max_length" in ds:
        args.max_length = ds["max_length"]

    # Model config
    model = config.get("model", {})
    if args.model_name == "google/electra-base-discriminator" and "name" in model:
        args.model_name = model["name"]
    if not args.use_spectral_norm and model.get("use_spectral_norm", False):
        args.use_spectral_norm = True

    # Training config
    train = config.get("training", {})
    if args.learning_rate == 5e-5 and "learning_rate" in train:
        args.learning_rate = train["learning_rate"]
    if args.per_device_train_batch_size == 32 and "per_device_train_batch_size" in train:
        args.per_device_train_batch_size = train["per_device_train_batch_size"]
    if args.num_train_epochs == 12 and "num_train_epochs" in train:
        args.num_train_epochs = train["num_train_epochs"]
    if args.weight_decay == 0.1 and "weight_decay" in train:
        args.weight_decay = train["weight_decay"]
    if args.warmup_ratio == 0.1 and "warmup_ratio" in train:
        args.warmup_ratio = train["warmup_ratio"]
    if args.seed == 42 and "seed" in train:
        args.seed = train["seed"]

    # Output config
    output = config.get("output", {})
    if args.cache_dir == "cache/encoder" and "cache_dir" in output:
        args.cache_dir = output["cache_dir"]


def get_variant_name(args) -> str:
    """Get a descriptive name for the model variant."""
    if args.use_spectral_norm:
        return "electra_mrpc_sn"
    return "electra_mrpc"


def get_cache_prefix(cache_dir: Path, variant: str, seed: int) -> str:
    """Get the cache file prefix for a given variant and seed."""
    return str(cache_dir / f"{variant}_seed{seed}")


def main():
    args = parse_args()
    logger = setup_logging()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        apply_config(args, config)

    set_seed(args.seed)

    variant = get_variant_name(args)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = get_cache_prefix(cache_dir, variant, args.seed)

    logger.info("=" * 70)
    logger.info("ELECTRA/MRPC Fine-tuning and Feature Extraction")
    logger.info("=" * 70)
    logger.info(f"Variant: {variant}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Spectral norm: {args.use_spectral_norm}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Warmup ratio: {args.warmup_ratio}")
    logger.info(f"n_cal: {args.n_cal}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Cache prefix: {cache_prefix}")

    # =========================================================================
    # Step 1: Load MRPC data
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Loading MRPC dataset")
    logger.info("=" * 70)

    data = load_mrpc(
        model_name=args.model_name,
        n_cal=args.n_cal,
        max_length=args.max_length,
        seed=args.seed,
    )

    train_dataset = data["train_dataset"]
    cal_dataset = data["cal_dataset"]
    test_dataset = data["test_dataset"]

    logger.info(f"Train: {data['n_train']} samples")
    logger.info(f"Cal: {data['n_cal']} samples")
    logger.info(f"Test: {data['n_test']} samples")

    # =========================================================================
    # Step 2: Fine-tune ELECTRA (or load from checkpoint)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Fine-tuning ELECTRA")
    logger.info("=" * 70)

    model_save_dir = cache_dir / f"{variant}_model_seed{args.seed}"

    if args.skip_training and model_save_dir.exists():
        logger.info(f"Loading existing model from {model_save_dir}")
        classifier = EncoderClassifier.load(
            path=str(model_save_dir),
            model_name=args.model_name,
            num_labels=args.num_labels,
            use_spectral_norm=args.use_spectral_norm,
        )
    else:
        classifier = EncoderClassifier(
            model_name=args.model_name,
            num_labels=args.num_labels,
            use_spectral_norm=args.use_spectral_norm,
        )

        checkpoint_dir = Path(args.checkpoint_dir) / f"{variant}_seed{args.seed}"

        # Train for fixed number of epochs (matching the paper).
        # We pass test_dataset for monitoring eval loss/accuracy in logs,
        # but do NOT select the best model based on it (load_best_model_at_end=False).
        classifier.finetune(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            output_dir=str(checkpoint_dir),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            load_best_model_at_end=False,
        )

        # Save the final model
        classifier.save(str(model_save_dir))
        logger.info(f"Model saved to {model_save_dir}")

    # =========================================================================
    # Step 3: Evaluate classification performance
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Evaluating classification performance")
    logger.info("=" * 70)

    # Predict on all splits
    logger.info("\nPredicting on training set...")
    train_probs, train_preds, train_labels = classifier.predict(train_dataset)

    logger.info("\nPredicting on calibration set...")
    cal_probs, cal_preds, cal_labels = classifier.predict(cal_dataset)

    logger.info("\nPredicting on test set...")
    test_probs, test_preds, test_labels = classifier.predict(test_dataset)

    # Report accuracy
    train_acc = (train_preds == train_labels).mean()
    cal_acc = (cal_preds == cal_labels).mean()
    test_acc = (test_preds == test_labels).mean()

    logger.info(f"\nClassification accuracy:")
    logger.info(f"  Train: {train_acc:.4f} ({(train_preds == train_labels).sum()}/{len(train_labels)})")
    logger.info(f"  Cal:   {cal_acc:.4f} ({(cal_preds == cal_labels).sum()}/{len(cal_labels)})")
    logger.info(f"  Test:  {test_acc:.4f} ({(test_preds == test_labels).sum()}/{len(test_labels)})")

    # Compute F1 on test set
    from sklearn.metrics import f1_score, classification_report
    test_f1 = f1_score(test_labels, test_preds)
    logger.info(f"  Test F1: {test_f1:.4f}")
    logger.info(f"\nTest set classification report:")
    logger.info("\n" + classification_report(
        test_labels, test_preds, target_names=["Not Paraphrase", "Paraphrase"]
    ))

    # =========================================================================
    # Step 4: Extract features from all splits
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Extracting penultimate-layer features")
    logger.info("=" * 70)

    logger.info("\nExtracting train features...")
    train_features, train_feat_labels = classifier.extract_features(train_dataset)

    logger.info("Extracting cal features...")
    cal_features, cal_feat_labels = classifier.extract_features(cal_dataset)

    logger.info("Extracting test features...")
    test_features, test_feat_labels = classifier.extract_features(test_dataset)

    # Verify label consistency
    assert np.array_equal(train_labels, train_feat_labels), "Train label mismatch"
    assert np.array_equal(cal_labels, cal_feat_labels), "Cal label mismatch"
    assert np.array_equal(test_labels, test_feat_labels), "Test label mismatch"

    logger.info(f"\nFeature shapes:")
    logger.info(f"  Train: {train_features.shape}")
    logger.info(f"  Cal:   {cal_features.shape}")
    logger.info(f"  Test:  {test_features.shape}")

    # =========================================================================
    # Step 5: Fit MahalanobisScorer on training features
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Fitting Mahalanobis distance scorer")
    logger.info("=" * 70)

    scorer = MahalanobisScorer(num_classes=args.num_labels)
    scorer.fit(train_features, train_labels)

    logger.info(f"Scorer stats: {scorer.get_stats()}")

    # =========================================================================
    # Step 6: Compute MD scores on all splits
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Computing Mahalanobis distance scores")
    logger.info("=" * 70)

    logger.info("\nScoring training set...")
    train_md_scores = scorer.score(train_features)

    logger.info("Scoring calibration set...")
    cal_md_scores = scorer.score(cal_features)

    logger.info("Scoring test set...")
    test_md_scores = scorer.score(test_features)

    # Report MD score statistics
    for name, scores in [("Train", train_md_scores), ("Cal", cal_md_scores),
                         ("Test", test_md_scores)]:
        logger.info(f"  {name} MD scores: min={scores.min():.4f}, "
                    f"max={scores.max():.4f}, mean={scores.mean():.4f}, "
                    f"std={scores.std():.4f}")

    # Quick discriminative check: do errors have higher MD scores?
    _, test_errors = get_predictions_and_errors(test_probs, test_labels)
    correct_md = test_md_scores[test_errors == 0]
    error_md = test_md_scores[test_errors == 1]
    if len(error_md) > 0:
        logger.info(f"\n  Test MD by correctness:")
        logger.info(f"    Correct: mean={correct_md.mean():.4f}, std={correct_md.std():.4f}")
        logger.info(f"    Error:   mean={error_md.mean():.4f}, std={error_md.std():.4f}")

    # =========================================================================
    # Step 7: Save everything to cache
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 7: Saving cached outputs")
    logger.info("=" * 70)

    # Save features + labels + probabilities + predictions for each split
    for split_name, features, labels, probs, preds, md_scores in [
        ("train", train_features, train_labels, train_probs, train_preds, train_md_scores),
        ("cal", cal_features, cal_labels, cal_probs, cal_preds, cal_md_scores),
        ("test", test_features, test_labels, test_probs, test_preds, test_md_scores),
    ]:
        npz_path = f"{cache_prefix}_{split_name}.npz"
        np.savez(
            npz_path,
            features=features,
            labels=labels,
            probs=probs,
            predictions=preds,
            md_scores=md_scores,
        )
        logger.info(f"  Saved {split_name} data to {npz_path}")

    # Save MD scorer
    scorer_path = f"{cache_prefix}_md_scorer.pkl"
    scorer.save(scorer_path)

    # Save experiment metadata
    metadata = {
        "variant": variant,
        "model_name": args.model_name,
        "use_spectral_norm": args.use_spectral_norm,
        "num_labels": args.num_labels,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "seed": args.seed,
        "n_train": data["n_train"],
        "n_cal": data["n_cal"],
        "n_test": data["n_test"],
        "train_accuracy": float(train_acc),
        "cal_accuracy": float(cal_acc),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "feature_dim": int(train_features.shape[1]),
        "scorer_stats": scorer.get_stats(),
        "cache_prefix": cache_prefix,
    }

    metadata_path = f"{cache_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata to {metadata_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Variant: {variant}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Spectral norm: {args.use_spectral_norm}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"Feature dim: {train_features.shape[1]}")
    logger.info(f"Train MD scores: {train_md_scores.mean():.4f} +/- {train_md_scores.std():.4f}")
    logger.info(f"Cal MD scores: {cal_md_scores.mean():.4f} +/- {cal_md_scores.std():.4f}")
    logger.info(f"Test MD scores: {test_md_scores.mean():.4f} +/- {test_md_scores.std():.4f}")
    logger.info(f"\nCached outputs at: {cache_prefix}_*.npz")
    logger.info(f"MD scorer at: {scorer_path}")
    logger.info(f"Metadata at: {metadata_path}")
    logger.info(f"\nNext step: run scripts/run_encoder_uncertainty.py "
                f"--cache_prefix {cache_prefix}")


if __name__ == "__main__":
    main()
