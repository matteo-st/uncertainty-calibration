#!/usr/bin/env python3
"""
Cache model outputs on unused training samples for n_cal ablation.

SST-2 and AG News were subsampled to 10%, leaving ~60K and ~108K unused
training samples. This script loads the fine-tuned models, runs inference
and feature extraction on those unused samples, and saves the results as
.npz files (same format as the train/cal/test splits).

These cached outputs are then used by run_ncal_ablation.py to evaluate
calibration methods at various n_cal values without re-running GPU inference.

Usage:
    python scripts/cache_unused_samples.py \
        --datasets sst2 agnews \
        --cache_dir cache/paper \
        --seeds 42 123 456
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging

from datasets import load_dataset
from src.encoder_data import _load_tokenizer
from src.encoder_models import EncoderClassifier
from src.mahalanobis import MahalanobisScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Model name -> short name (must match finetune_encoder.py)
MODEL_SHORT_NAMES = {
    "google/electra-base-discriminator": "electra",
    "google-bert/bert-base-uncased": "bert",
    "microsoft/deberta-v3-base": "deberta",
}

# Dataset configs: how the original 10% subsample was carved
DATASET_CONFIGS = {
    "sst2": {
        "hf_path": "glue",
        "hf_name": "sst2",
        "split": "train",
        "text_key": "sentence",
        "pair_key": None,
        "n_total": 67349,
        "n_train": 5735,
        "n_cal": 1000,
        "n_test": 7600,
        "num_labels": 2,
        "max_length": 128,
    },
    "agnews": {
        "hf_path": "ag_news",
        "hf_name": None,
        "split": "train",
        "text_key": "text",
        "pair_key": None,
        "n_total": 120000,
        "n_train": 11000,
        "n_cal": 1000,
        "num_labels": 4,
        "max_length": 128,
    },
}

MODELS = [
    "google/electra-base-discriminator",
    "google-bert/bert-base-uncased",
    "microsoft/deberta-v3-base",
]


def get_unused_dataset(dataset_name, model_name, seed, max_length=128):
    """
    Load the unused portion of the training data (samples not in train or cal).

    Reconstructs the same shuffle used by encoder_data._split_train_cal,
    then selects the indices beyond train + cal.

    Returns:
        unused_dataset: HuggingFace Dataset (tokenized, torch format)
        n_unused: number of unused samples
    """
    cfg = DATASET_CONFIGS[dataset_name]

    # Load raw dataset
    if cfg["hf_name"]:
        raw = load_dataset(cfg["hf_path"], cfg["hf_name"])
    else:
        raw = load_dataset(cfg["hf_path"])
    full_train = raw[cfg["split"]]

    n_total = len(full_train)
    assert n_total == cfg["n_total"], (
        f"Expected {cfg['n_total']} samples, got {n_total}"
    )

    # Reconstruct the same shuffle as encoder_data._split_train_cal
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n_total).tolist()

    # Unused = everything after train + cal + test (if test carved from pool)
    n_test = cfg.get("n_test", 0)
    start = cfg["n_train"] + cfg["n_cal"] + n_test
    unused_indices = shuffled_indices[start:]
    n_unused = len(unused_indices)
    logger.info(
        f"{dataset_name} seed={seed}: {n_total} total, "
        f"train={cfg['n_train']}, cal={cfg['n_cal']}, "
        f"test={n_test}, unused={n_unused}"
    )

    # Select unused subset
    unused_dataset = full_train.select(unused_indices)

    # Tokenize
    tokenizer = _load_tokenizer(model_name)

    if cfg["pair_key"]:
        def tokenize_fn(examples):
            return tokenizer(
                examples[cfg["text_key"]],
                examples[cfg["pair_key"]],
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
    else:
        def tokenize_fn(examples):
            return tokenizer(
                examples[cfg["text_key"]],
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

    unused_dataset = unused_dataset.map(tokenize_fn, batched=True)

    # Set torch format
    columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
    available = [c for c in columns if c in unused_dataset.column_names]
    unused_dataset.set_format("torch", columns=available)

    return unused_dataset, n_unused


def process_one(dataset_name, model_name, seed, cache_dir):
    """Process one (dataset, model, seed) combination."""
    cfg = DATASET_CONFIGS[dataset_name]
    model_short = MODEL_SHORT_NAMES[model_name]
    variant = f"{model_short}_{dataset_name}"
    cache_prefix = str(cache_dir / f"{variant}_seed{seed}")

    # Check if output already exists
    out_path = f"{cache_prefix}_unused.npz"
    if Path(out_path).exists():
        logger.info(f"SKIP (already exists): {out_path}")
        return

    # Load unused data
    unused_dataset, n_unused = get_unused_dataset(
        dataset_name, model_name, seed, cfg["max_length"]
    )

    # Load fine-tuned model
    model_dir = cache_dir / f"{variant}_model_seed{seed}"
    logger.info(f"Loading model from {model_dir}")
    classifier = EncoderClassifier.load(
        path=str(model_dir),
        model_name=model_name,
        num_labels=cfg["num_labels"],
        use_spectral_norm=False,
    )

    # Run prediction
    logger.info(f"Running prediction on {n_unused} unused samples...")
    probs, predictions, labels, logits = classifier.predict(unused_dataset)

    # Extract features
    logger.info(f"Extracting features from {n_unused} unused samples...")
    features, feat_labels = classifier.extract_features(unused_dataset)
    assert np.array_equal(labels, feat_labels), "Label mismatch after feature extraction"

    # Compute MD scores
    scorer_path = f"{cache_prefix}_md_scorer.pkl"
    logger.info(f"Loading MD scorer from {scorer_path}")
    scorer = MahalanobisScorer.load(scorer_path)
    md_scores = scorer.score(features)

    # Save
    np.savez(
        out_path,
        features=features,
        labels=labels,
        probs=probs,
        logits=logits,
        predictions=predictions,
        md_scores=md_scores,
    )
    logger.info(
        f"Saved {out_path}: {n_unused} samples, "
        f"features={features.shape}, md_scores={md_scores.shape}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cache model outputs on unused training samples"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["sst2", "agnews"],
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--cache_dir", type=str, default="cache/paper")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    total = len(args.datasets) * len(MODELS) * len(args.seeds)
    done = 0

    for dataset_name in args.datasets:
        for model_name in MODELS:
            for seed in args.seeds:
                done += 1
                model_short = MODEL_SHORT_NAMES[model_name]
                logger.info(
                    f"\n{'='*70}\n"
                    f"[{done}/{total}] {dataset_name} / {model_short} / seed={seed}\n"
                    f"{'='*70}"
                )
                process_one(dataset_name, model_name, seed, cache_dir)

    logger.info(f"\nAll done! Processed {total} combinations.")


if __name__ == "__main__":
    main()
