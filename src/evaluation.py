"""
Evaluation metrics for calibration.

Includes accuracy, cross-entropy, and normalized cross-entropy
as used in the paper.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation results."""
    accuracy: float
    error_rate: float
    cross_entropy: float
    normalized_cross_entropy: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "cross_entropy": self.cross_entropy,
            "normalized_cross_entropy": self.normalized_cross_entropy,
        }


def compute_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)

    Returns:
        Accuracy (0 to 1)
    """
    predictions = probs.argmax(axis=1)
    return (predictions == labels).mean()


def compute_error_rate(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute error rate (1 - accuracy)."""
    return 1.0 - compute_accuracy(probs, labels)


def compute_cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute cross-entropy loss.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)

    Returns:
        Mean cross-entropy
    """
    n_samples = len(labels)
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(probs, 1e-10, 1.0)

    # Get probability of true class for each sample
    true_probs = probs_clipped[np.arange(n_samples), labels]

    return -np.log(true_probs).mean()


def compute_normalized_cross_entropy(
    probs: np.ndarray,
    labels: np.ndarray,
    prior: Optional[np.ndarray] = None,
) -> float:
    """
    Compute normalized cross-entropy.

    Normalized by the cross-entropy of a naive classifier that always
    outputs the prior distribution.

    A value > 1.0 means the classifier is worse than the naive baseline.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        prior: Prior distribution (default: empirical from labels)

    Returns:
        Normalized cross-entropy
    """
    ce = compute_cross_entropy(probs, labels)

    # Compute baseline cross-entropy
    if prior is None:
        n_classes = probs.shape[1]
        label_counts = np.bincount(labels, minlength=n_classes)
        prior = label_counts / len(labels)

    # Baseline always predicts prior
    prior_clipped = np.clip(prior, 1e-10, 1.0)
    baseline_ce = -np.log(prior_clipped[labels]).mean()

    return ce / baseline_ce


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    prior: Optional[np.ndarray] = None,
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        prior: Prior distribution for normalized CE (default: empirical)

    Returns:
        EvaluationMetrics object
    """
    return EvaluationMetrics(
        accuracy=compute_accuracy(probs, labels),
        error_rate=compute_error_rate(probs, labels),
        cross_entropy=compute_cross_entropy(probs, labels),
        normalized_cross_entropy=compute_normalized_cross_entropy(probs, labels, prior),
    )


def bootstrap_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics with bootstrap confidence intervals.

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Dict with mean and std for each metric
    """
    rng = np.random.RandomState(seed)
    n_samples = len(labels)

    metrics_list = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, n_samples, replace=True)
        metrics = compute_metrics(probs[indices], labels[indices])
        metrics_list.append(metrics.to_dict())

    # Aggregate
    result = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        result[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    return result
