"""
Dataset loading for encoder-based classification experiments.

Handles MRPC (and potentially other GLUE tasks) with train/cal/test splits.
Unlike data.py (prompt-based for decoder LMs), this module produces
tokenized input pairs for standard sequence classification.
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_mrpc(
    model_name: str = "google/electra-base-discriminator",
    n_cal: int = 600,
    max_length: int = 128,
    seed: int = 42,
) -> Dict:
    """
    Load MRPC dataset with train/cal/test splits for encoder classification.

    The MRPC training set (3,668 samples) is split into:
    - Train: first (3668 - n_cal) samples after shuffling -- for fine-tuning
    - Cal: last n_cal samples after shuffling -- for score calibration
    The GLUE validation set (408 samples) is used as the test set
    (standard practice since GLUE test labels are private).

    Shuffling uses a fixed seed for reproducibility. The split is deterministic.

    Args:
        model_name: HuggingFace model name (determines tokenizer)
        n_cal: Number of calibration samples (reserved from training set)
        max_length: Maximum sequence length for tokenization
        seed: Random seed for shuffling before splitting

    Returns:
        Dict with keys:
            - train_dataset: HuggingFace Dataset for fine-tuning
            - cal_dataset: HuggingFace Dataset for score calibration
            - test_dataset: HuggingFace Dataset for evaluation
            - tokenizer: Pre-trained tokenizer
            - n_train: Actual number of training samples
            - n_cal: Actual number of calibration samples
            - n_test: Actual number of test samples
            - num_labels: Number of classes (2 for MRPC)
    """
    logger.info("Loading MRPC dataset from GLUE...")
    raw_datasets = load_dataset("glue", "mrpc")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get full training set
    full_train = raw_datasets["train"]
    n_total_train = len(full_train)
    n_train = n_total_train - n_cal

    logger.info(f"MRPC full training set: {n_total_train} samples")
    logger.info(f"Split: train={n_train}, cal={n_cal}")
    logger.info(f"Test set (GLUE validation): {len(raw_datasets['validation'])} samples")

    if n_cal >= n_total_train:
        raise ValueError(
            f"n_cal={n_cal} must be less than total training size={n_total_train}"
        )

    # Shuffle training set with fixed seed, then split
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n_total_train).tolist()

    train_indices = shuffled_indices[:n_train]
    cal_indices = shuffled_indices[n_train:]

    train_dataset = full_train.select(train_indices)
    cal_dataset = full_train.select(cal_indices)
    test_dataset = raw_datasets["validation"]

    # Tokenize all splits
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    cal_dataset = cal_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # Set format for PyTorch
    columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
    # Some tokenizers may not produce token_type_ids
    available_columns = [
        c for c in columns if c in train_dataset.column_names
    ]
    train_dataset.set_format("torch", columns=available_columns)
    cal_dataset.set_format("torch", columns=available_columns)
    test_dataset.set_format("torch", columns=available_columns)

    logger.info(f"Tokenized with max_length={max_length}")
    logger.info(f"Final sizes: train={len(train_dataset)}, "
                f"cal={len(cal_dataset)}, test={len(test_dataset)}")

    # Report label distribution
    for split_name, ds in [("train", train_dataset), ("cal", cal_dataset),
                           ("test", test_dataset)]:
        labels = np.array(ds["label"])
        counts = np.bincount(labels, minlength=2)
        logger.info(f"  {split_name} label distribution: "
                    f"0={counts[0]} ({counts[0]/len(labels)*100:.1f}%), "
                    f"1={counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

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
