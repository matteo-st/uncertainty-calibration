"""
Dataset loading for encoder-based classification experiments.

Supports GLUE tasks (MRPC, SST2, CoLA) and AG News with train/cal/test splits.
Unlike data.py (prompt-based for decoder LMs), this module produces
tokenized inputs for standard sequence classification.

Usage:
    # Generic dispatcher (preferred)
    data = load_encoder_dataset("mrpc", n_cal=600)
    data = load_encoder_dataset("sst2", n_train=5735, n_cal=1000)
    data = load_encoder_dataset("cola", n_cal=1000)
    data = load_encoder_dataset("agnews", n_train=11000, n_cal=1000)

    # Backward-compatible wrapper
    data = load_mrpc(n_cal=600)
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# DeBERTa-v3 SentencePiece tokenizer breaks with the fast tokenizer
# conversion path in transformers >=5.x (tiktoken parsing error).
_SLOW_TOKENIZER_MODELS = {"microsoft/deberta-v3-base"}


def _load_tokenizer(model_name: str):
    """Load tokenizer, falling back to slow tokenizer for known-broken models."""
    use_fast = model_name not in _SLOW_TOKENIZER_MODELS
    return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)


def load_encoder_dataset(
    dataset_name: str,
    model_name: str = "google/electra-base-discriminator",
    n_train: Optional[int] = None,
    n_cal: int = 600,
    max_length: int = 128,
    seed: int = 42,
) -> Dict:
    """
    Load a dataset for encoder classification experiments.

    Supports MRPC (sentence pair) and SST2 (single sentence).
    Handles train/cal/test splitting, tokenization, and format setup.

    Args:
        dataset_name: Dataset identifier ("mrpc" or "sst2")
        model_name: HuggingFace model name (determines tokenizer)
        n_train: Explicit train size. If None, uses all available minus n_cal.
        n_cal: Number of calibration samples (reserved from training set)
        max_length: Maximum sequence length for tokenization
        seed: Random seed for shuffling before splitting

    Returns:
        Dict with keys:
            - train_dataset: HuggingFace Dataset for fine-tuning
            - cal_dataset: HuggingFace Dataset for score calibration
            - test_dataset: HuggingFace Dataset for evaluation
            - tokenizer: Pre-trained tokenizer
            - n_train, n_cal, n_test: Actual split sizes
            - num_labels: Number of classes
    """
    loaders = {
        "mrpc": _load_mrpc,
        "sst2": _load_sst2,
        "cola": _load_cola,
        "agnews": _load_agnews,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: {list(loaders.keys())}")
    return loaders[dataset_name](model_name, n_train, n_cal, max_length, seed)


def load_mrpc(
    model_name: str = "google/electra-base-discriminator",
    n_cal: int = 600,
    max_length: int = 128,
    seed: int = 42,
) -> Dict:
    """Backward-compatible wrapper. Delegates to load_encoder_dataset("mrpc", ...)."""
    return load_encoder_dataset("mrpc", model_name=model_name, n_cal=n_cal,
                                max_length=max_length, seed=seed)


def _load_mrpc(model_name, n_train, n_cal, max_length, seed) -> Dict:
    """
    Load MRPC dataset with train/cal/test splits.

    MRPC is a sentence-pair task (paraphrase detection, 2 classes).
    Train set: 3,668 samples. Test set: GLUE validation (408 samples).
    """
    logger.info("Loading MRPC dataset from GLUE...")
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = _load_tokenizer(model_name)

    full_train = raw_datasets["train"]
    test_dataset = raw_datasets["validation"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    train_dataset, cal_dataset = _split_train_cal(
        full_train, n_train, n_cal, seed, dataset_name="MRPC"
    )

    train_dataset, cal_dataset, test_dataset = _finalize_splits(
        train_dataset, cal_dataset, test_dataset, tokenize_fn
    )

    return {
        "train_dataset": train_dataset,
        "cal_dataset": cal_dataset,
        "test_dataset": test_dataset,
        "tokenizer": tokenizer,
        "n_train": len(train_dataset),
        "n_cal": len(cal_dataset),
        "n_test": len(test_dataset),
        "num_labels": 2,
    }


def _load_sst2(model_name, n_train, n_cal, max_length, seed) -> Dict:
    """
    Load SST2 dataset with train/cal/test splits.

    SST2 is a single-sentence task (binary sentiment, 2 classes).
    Train set: 67,349 samples. Test set: GLUE validation (872 samples).
    """
    logger.info("Loading SST2 dataset from GLUE...")
    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = _load_tokenizer(model_name)

    full_train = raw_datasets["train"]
    test_dataset = raw_datasets["validation"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    train_dataset, cal_dataset = _split_train_cal(
        full_train, n_train, n_cal, seed, dataset_name="SST2"
    )

    train_dataset, cal_dataset, test_dataset = _finalize_splits(
        train_dataset, cal_dataset, test_dataset, tokenize_fn
    )

    return {
        "train_dataset": train_dataset,
        "cal_dataset": cal_dataset,
        "test_dataset": test_dataset,
        "tokenizer": tokenizer,
        "n_train": len(train_dataset),
        "n_cal": len(cal_dataset),
        "n_test": len(test_dataset),
        "num_labels": 2,
    }


def _load_cola(model_name, n_train, n_cal, max_length, seed) -> Dict:
    """
    Load CoLA dataset with train/cal/test splits.

    CoLA is a single-sentence task (linguistic acceptability, 2 classes).
    Train set: 8,551 samples. Test set: GLUE validation (1,043 samples).
    """
    logger.info("Loading CoLA dataset from GLUE...")
    raw_datasets = load_dataset("glue", "cola")
    tokenizer = _load_tokenizer(model_name)

    full_train = raw_datasets["train"]
    test_dataset = raw_datasets["validation"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    train_dataset, cal_dataset = _split_train_cal(
        full_train, n_train, n_cal, seed, dataset_name="CoLA"
    )

    train_dataset, cal_dataset, test_dataset = _finalize_splits(
        train_dataset, cal_dataset, test_dataset, tokenize_fn
    )

    return {
        "train_dataset": train_dataset,
        "cal_dataset": cal_dataset,
        "test_dataset": test_dataset,
        "tokenizer": tokenizer,
        "n_train": len(train_dataset),
        "n_cal": len(cal_dataset),
        "n_test": len(test_dataset),
        "num_labels": 2,
    }


def _load_agnews(model_name, n_train, n_cal, max_length, seed) -> Dict:
    """
    Load AG News dataset with train/cal/test splits.

    AG News is a single-sentence topic classification task (4 classes).
    Train set: 120,000 samples. Test set: 7,600 samples (own test split).
    """
    logger.info("Loading AG News dataset...")
    raw_datasets = load_dataset("ag_news")
    tokenizer = _load_tokenizer(model_name)

    full_train = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    train_dataset, cal_dataset = _split_train_cal(
        full_train, n_train, n_cal, seed, dataset_name="AG News"
    )

    train_dataset, cal_dataset, test_dataset = _finalize_splits(
        train_dataset, cal_dataset, test_dataset, tokenize_fn
    )

    return {
        "train_dataset": train_dataset,
        "cal_dataset": cal_dataset,
        "test_dataset": test_dataset,
        "tokenizer": tokenizer,
        "n_train": len(train_dataset),
        "n_cal": len(cal_dataset),
        "n_test": len(test_dataset),
        "num_labels": 4,
    }


def _split_train_cal(full_train, n_train, n_cal, seed, dataset_name):
    """
    Split a full training set into train and calibration subsets.

    Shuffles with a fixed seed, then selects:
    - If n_train is None: train = all except last n_cal, cal = last n_cal
    - If n_train is set: train = first n_train, cal = next n_cal

    Args:
        full_train: Full HuggingFace training Dataset
        n_train: Explicit train size (None = use all available minus n_cal)
        n_cal: Calibration set size
        seed: Random seed for shuffling
        dataset_name: Name for logging

    Returns:
        (train_dataset, cal_dataset)
    """
    n_total = len(full_train)

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n_total).tolist()

    if n_train is not None:
        if n_train + n_cal > n_total:
            raise ValueError(
                f"n_train={n_train} + n_cal={n_cal} = {n_train + n_cal} "
                f"> total training size={n_total}"
            )
        train_indices = shuffled_indices[:n_train]
        cal_indices = shuffled_indices[n_train:n_train + n_cal]
        n_reserved = n_total - n_train - n_cal
    else:
        if n_cal >= n_total:
            raise ValueError(
                f"n_cal={n_cal} must be less than total training size={n_total}"
            )
        n_train = n_total - n_cal
        train_indices = shuffled_indices[:n_train]
        cal_indices = shuffled_indices[n_train:]
        n_reserved = 0

    logger.info(f"{dataset_name} full training set: {n_total} samples")
    logger.info(f"Split: train={n_train}, cal={n_cal}"
                + (f", reserved={n_reserved}" if n_reserved > 0 else ""))

    train_dataset = full_train.select(train_indices)
    cal_dataset = full_train.select(cal_indices)

    return train_dataset, cal_dataset


def _finalize_splits(train_dataset, cal_dataset, test_dataset, tokenize_fn):
    """
    Tokenize all splits, set PyTorch format, and report label distributions.

    Args:
        train_dataset, cal_dataset, test_dataset: HuggingFace Datasets (raw)
        tokenize_fn: Tokenization function (dataset-specific)

    Returns:
        (train_dataset, cal_dataset, test_dataset) -- tokenized and formatted
    """
    # Tokenize
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    cal_dataset = cal_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # Set format for PyTorch
    columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
    available_columns = [c for c in columns if c in train_dataset.column_names]
    train_dataset.set_format("torch", columns=available_columns)
    cal_dataset.set_format("torch", columns=available_columns)
    test_dataset.set_format("torch", columns=available_columns)

    logger.info(f"Final sizes: train={len(train_dataset)}, "
                f"cal={len(cal_dataset)}, test={len(test_dataset)}")

    # Report label distribution
    for split_name, ds in [("train", train_dataset), ("cal", cal_dataset),
                           ("test", test_dataset)]:
        labels = np.array(ds["label"])
        counts = np.bincount(labels)
        dist_parts = [f"{i}={c} ({c/len(labels)*100:.1f}%)"
                      for i, c in enumerate(counts)]
        logger.info(f"  {split_name} label distribution: {', '.join(dist_parts)}")

    logger.info(f"Test set: {len(test_dataset)} samples")

    return train_dataset, cal_dataset, test_dataset
