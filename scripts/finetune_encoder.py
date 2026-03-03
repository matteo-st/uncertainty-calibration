#!/usr/bin/env python3
"""
Fine-tune an encoder model and extract features for Mahalanobis distance.

Supports multiple datasets (MRPC, SST2) and model variants (MD, MD SN).

Pipeline:
1. Load dataset with train/cal/test splits
2. Fine-tune encoder on the training split
3. Evaluate classification accuracy on all splits
4. Extract penultimate-layer features from all splits
5. Fit MahalanobisScorer on training features
6. Compute MD scores on calibration and test sets
7. Save everything to cache (model, features, scorer, scores, probabilities)

Usage:
    # MRPC - Standard MD variant
    python finetune_encoder.py --config configs/mrpc_electra.yaml

    # MRPC - MD SN variant
    python finetune_encoder.py --config configs/mrpc_electra.yaml \
        --use_spectral_norm --learning_rate 3e-5 --num_train_epochs 11

    # SST2 - Standard MD variant
    python finetune_encoder.py --config configs/sst2_electra.yaml

    # SST2 - MD SN variant
    python finetune_encoder.py --config configs/sst2_electra.yaml \
        --use_spectral_norm --learning_rate 5e-5 --num_train_epochs 7 \
        --weight_decay 0.01

    # Quick test with small data
    python finetune_encoder.py --dataset sst2 --n_train 200 --n_cal 50 \
        --num_train_epochs 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import logging

from src.encoder_data import load_encoder_dataset
from src.encoder_models import EncoderClassifier
from src.mahalanobis import MahalanobisScorer
from src.uncertainty import compute_uncertainty_scores, get_predictions_and_errors
from src.utils import set_seed, setup_logging, load_config


# Dataset-specific classification report labels
DATASET_LABELS = {
    "mrpc": ["Not Paraphrase", "Paraphrase"],
    "sst2": ["Negative", "Positive"],
    "cola": ["Unacceptable", "Acceptable"],
    "agnews": ["World", "Sports", "Business", "Sci/Tech"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune encoder model and extract MD features"
    )

    # Configuration file (optional, CLI args override config)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Dataset
    parser.add_argument("--dataset", type=str, default="mrpc",
                        choices=["mrpc", "sst2", "cola", "agnews"],
                        help="Dataset to fine-tune on")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Explicit train size (None = use all available minus n_cal)")

    # Model
    parser.add_argument("--model_name", type=str,
                        default="google/electra-base-discriminator")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--use_spectral_norm", action="store_true",
                        help="Apply spectral normalization (MD SN variant)")

    # Data
    parser.add_argument("--n_cal", type=int, default=600,
                        help="Number of calibration samples from train set")
    parser.add_argument("--n_test", type=int, default=None,
                        help="Test set size carved from training pool (default: use external test set)")
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

    # Track which args were explicitly provided on the CLI
    # so apply_config() can avoid overriding them.
    args = parser.parse_args()
    defaults = parser.parse_args([])
    cli_args = set()
    for key in vars(args):
        if getattr(args, key) != getattr(defaults, key):
            cli_args.add(key)
    args._cli_args = cli_args
    return args


def apply_config(args, config: dict, cli_args: set) -> None:
    """
    Apply YAML config values to args, without overriding CLI-provided values.

    Args:
        args: Parsed argparse namespace
        config: YAML config dict
        cli_args: Set of arg names explicitly provided on the command line
    """
    def _set(attr, value):
        """Set attr from config only if not explicitly provided on CLI."""
        if attr not in cli_args:
            setattr(args, attr, value)

    # Dataset config
    ds = config.get("dataset", {})
    if "name" in ds:
        _set("dataset", ds["name"])
    if "n_train" in ds:
        _set("n_train", ds["n_train"])
    if "n_cal" in ds:
        _set("n_cal", ds["n_cal"])
    if "n_test" in ds:
        _set("n_test", ds["n_test"])
    if "max_length" in ds:
        _set("max_length", ds["max_length"])

    # Model config
    model = config.get("model", {})
    if "name" in model:
        _set("model_name", model["name"])
    if "use_spectral_norm" in model:
        _set("use_spectral_norm", model["use_spectral_norm"])

    # Training config
    train = config.get("training", {})
    if "learning_rate" in train:
        _set("learning_rate", train["learning_rate"])
    if "per_device_train_batch_size" in train:
        _set("per_device_train_batch_size", train["per_device_train_batch_size"])
    if "num_train_epochs" in train:
        _set("num_train_epochs", train["num_train_epochs"])
    if "weight_decay" in train:
        _set("weight_decay", train["weight_decay"])
    if "warmup_ratio" in train:
        _set("warmup_ratio", train["warmup_ratio"])
    if "seed" in train:
        _set("seed", train["seed"])

    # Output config
    output = config.get("output", {})
    if "cache_dir" in output:
        _set("cache_dir", output["cache_dir"])


MODEL_SHORT_NAMES = {
    "google/electra-base-discriminator": "electra",
    "microsoft/deberta-v3-base": "deberta",
    "google-bert/bert-base-uncased": "bert",
}


def get_variant_name(args) -> str:
    """Get a descriptive name for the model variant (includes dataset)."""
    model_short = MODEL_SHORT_NAMES.get(args.model_name, args.model_name.split("/")[-1])
    base = f"{model_short}_{args.dataset}"
    if args.use_spectral_norm:
        return f"{base}_sn"
    return base


def get_cache_prefix(cache_dir: Path, variant: str, seed: int) -> str:
    """Get the cache file prefix for a given variant and seed."""
    return str(cache_dir / f"{variant}_seed{seed}")


def main():
    args = parse_args()
    logger = setup_logging()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        apply_config(args, config, getattr(args, '_cli_args', set()))

    set_seed(args.seed)

    variant = get_variant_name(args)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = get_cache_prefix(cache_dir, variant, args.seed)

    logger.info("=" * 70)
    logger.info(f"Encoder Fine-tuning and Feature Extraction")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Variant: {variant}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Spectral norm: {args.use_spectral_norm}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Warmup ratio: {args.warmup_ratio}")
    logger.info(f"n_train: {args.n_train if args.n_train else 'all available'}")
    logger.info(f"n_cal: {args.n_cal}")
    logger.info(f"n_test: {args.n_test if args.n_test else 'external test set'}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Cache prefix: {cache_prefix}")

    # =========================================================================
    # Step 1: Load dataset
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"Step 1: Loading {args.dataset.upper()} dataset")
    logger.info("=" * 70)

    data = load_encoder_dataset(
        dataset_name=args.dataset,
        model_name=args.model_name,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        max_length=args.max_length,
        seed=args.seed,
    )

    train_dataset = data["train_dataset"]
    cal_dataset = data["cal_dataset"]
    test_dataset = data["test_dataset"]

    # Override num_labels from dataset (handles multi-class like AG News)
    args.num_labels = data["num_labels"]

    logger.info(f"Train: {data['n_train']} samples")
    logger.info(f"Cal: {data['n_cal']} samples")
    logger.info(f"Test: {data['n_test']} samples")
    logger.info(f"Num labels: {args.num_labels}")

    # =========================================================================
    # Step 2: Fine-tune encoder (or load from checkpoint)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Fine-tuning encoder")
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
    train_probs, train_preds, train_labels, train_logits = classifier.predict(train_dataset)

    logger.info("\nPredicting on calibration set...")
    cal_probs, cal_preds, cal_labels, cal_logits = classifier.predict(cal_dataset)

    logger.info("\nPredicting on test set...")
    test_probs, test_preds, test_labels, test_logits = classifier.predict(test_dataset)

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
    f1_avg = "binary" if args.num_labels == 2 else "weighted"
    test_f1 = f1_score(test_labels, test_preds, average=f1_avg)
    logger.info(f"  Test F1 ({f1_avg}): {test_f1:.4f}")
    logger.info(f"\nTest set classification report:")
    target_names = DATASET_LABELS.get(args.dataset)
    logger.info("\n" + classification_report(
        test_labels, test_preds, target_names=target_names
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

    # Save features + labels + probabilities + logits + predictions for each split
    for split_name, features, labels, probs, logits, preds, md_scores in [
        ("train", train_features, train_labels, train_probs, train_logits, train_preds, train_md_scores),
        ("cal", cal_features, cal_labels, cal_probs, cal_logits, cal_preds, cal_md_scores),
        ("test", test_features, test_labels, test_probs, test_logits, test_preds, test_md_scores),
    ]:
        npz_path = f"{cache_prefix}_{split_name}.npz"
        np.savez(
            npz_path,
            features=features,
            labels=labels,
            probs=probs,
            logits=logits,
            predictions=preds,
            md_scores=md_scores,
        )
        logger.info(f"  Saved {split_name} data to {npz_path}")

    # Save MD scorer
    scorer_path = f"{cache_prefix}_md_scorer.pkl"
    scorer.save(scorer_path)

    # Save experiment metadata
    metadata = {
        "dataset": args.dataset,
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
    logger.info(f"Dataset: {args.dataset}")
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
