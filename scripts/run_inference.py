#!/usr/bin/env python3
"""
Run LLM inference to extract probabilities for all samples.

This script extracts P(y|q,e) from the LLM for train and test samples,
saving the raw probabilities for later calibration experiments.

Usage:
    python scripts/run_inference.py --dataset agnews --model gpt2-xl --n_shots 0
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LLMClassifier
from src.data import load_dataset_by_name
from src.utils import save_probabilities, set_seed, setup_logging
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM inference for classification")
    parser.add_argument("--dataset", type=str, default="agnews",
                        choices=["agnews", "sst2", "trec", "dbpedia"])
    parser.add_argument("--model", type=str, default="gpt2-xl")
    parser.add_argument("--n_shots", type=int, default=0,
                        help="Number of few-shot examples in prompt")
    parser.add_argument("--n_train", type=int, default=600,
                        help="Number of training samples")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/probabilities")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    set_seed(args.seed)

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    dataset = load_dataset_by_name(
        args.dataset,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )
    logger.info(f"  Train samples: {len(dataset.train_texts)}")
    logger.info(f"  Test samples: {len(dataset.test_texts)}")
    logger.info(f"  Classes: {dataset.label_names}")

    # Load model
    logger.info(f"Loading model {args.model}...")
    classifier = LLMClassifier(model_name=args.model)

    # Build preface with n_shots examples
    # Reserve first n_shots samples for the prompt, use rest for calibration
    if args.n_shots > 0:
        shot_indices = list(range(args.n_shots))
        train_indices = list(range(args.n_shots, len(dataset.train_texts)))
    else:
        shot_indices = []
        train_indices = list(range(len(dataset.train_texts)))

    preface = dataset.get_few_shot_preface(args.n_shots, shot_indices=shot_indices)
    logger.info(f"Preface ({args.n_shots}-shot):\n{preface[:500]}...")

    # Get probabilities for training samples (for calibration)
    logger.info("Computing probabilities for training samples...")
    train_prompts = [
        dataset.build_prompt(preface, dataset.train_texts[i])
        for i in train_indices
    ]
    train_probs = classifier.get_batch_label_probabilities(
        train_prompts,
        dataset.label_names,
        batch_size=args.batch_size,
    )
    train_labels = np.array([dataset.train_labels[i] for i in train_indices])

    # Get probabilities for test samples
    logger.info("Computing probabilities for test samples...")
    test_prompts = dataset.build_prompts_for_split(preface, split="test")
    test_probs = classifier.get_batch_label_probabilities(
        test_prompts,
        dataset.label_names,
        batch_size=args.batch_size,
    )
    test_labels = np.array(dataset.test_labels)

    # Get content-free probabilities
    logger.info("Computing content-free probabilities...")
    content_free_inputs = ["[MASK]", "N/A", ""]
    content_free_prompts = [
        dataset.build_prompt(preface, cf_input)
        for cf_input in content_free_inputs
    ]
    content_free_probs = classifier.get_batch_label_probabilities(
        content_free_prompts,
        dataset.label_names,
        batch_size=args.batch_size,
        show_progress=False,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{args.dataset}_{args.model.replace('/', '_')}_{args.n_shots}shot_seed{args.seed}"

    save_probabilities(
        train_probs, train_labels,
        output_dir / f"{output_name}_train.npz",
        metadata={"n_shots": args.n_shots, "dataset": args.dataset, "model": args.model},
    )
    save_probabilities(
        test_probs, test_labels,
        output_dir / f"{output_name}_test.npz",
        metadata={"n_shots": args.n_shots, "dataset": args.dataset, "model": args.model},
    )
    np.savez(
        output_dir / f"{output_name}_contentfree.npz",
        probs=content_free_probs,
        inputs=content_free_inputs,
    )

    logger.info(f"Results saved to {output_dir}")

    # Print summary statistics
    logger.info("\n=== Summary ===")
    logger.info(f"Train probs shape: {train_probs.shape}")
    logger.info(f"Test probs shape: {test_probs.shape}")
    logger.info(f"Content-free probs:\n{content_free_probs}")

    # Quick baseline accuracy
    baseline_acc = (test_probs.argmax(axis=1) == test_labels).mean()
    logger.info(f"Baseline (no calibration) test accuracy: {baseline_acc:.4f}")


if __name__ == "__main__":
    main()
