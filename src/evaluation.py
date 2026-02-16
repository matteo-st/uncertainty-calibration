"""
Evaluation metrics for calibration.

Includes:
- Classification metrics: accuracy, cross-entropy, normalized cross-entropy
- Error prediction metrics: ROCAUC, ECE, binary cross-entropy
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score


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


# =============================================================================
# Error Prediction Metrics (for uncertainty score calibration)
# =============================================================================

@dataclass
class ErrorPredictionMetrics:
    """Container for error prediction evaluation results."""
    rocauc: float
    ece: float
    binary_cross_entropy: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rocauc": self.rocauc,
            "ece": self.ece,
            "binary_cross_entropy": self.binary_cross_entropy,
        }


def compute_rocauc(
    scores: np.ndarray,
    errors: np.ndarray
) -> float:
    """
    Compute ROCAUC for error prediction.

    Higher score should correspond to higher probability of error.

    Args:
        scores: (N,) array of uncertainty scores (higher = more uncertain)
        errors: (N,) array of binary error indicators (1=error, 0=correct)

    Returns:
        ROCAUC score (0.5 = random, 1.0 = perfect)
    """
    # Handle edge case: all same class
    if len(np.unique(errors)) < 2:
        return 0.5

    return roc_auc_score(errors, scores)


def compute_ece(
    probs: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and
    actual error rates within bins.

    Args:
        probs: (N,) array of predicted error probabilities
        errors: (N,) array of binary error indicators (1=error, 0=correct)
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    n_samples = len(errors)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        # Find samples in this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            # Include right edge for last bin
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_count = in_bin.sum()

        if bin_count > 0:
            # Average predicted probability in bin
            avg_prob = probs[in_bin].mean()
            # Actual error rate in bin
            actual_rate = errors[in_bin].mean()
            # Weighted absolute difference
            ece += (bin_count / n_samples) * abs(avg_prob - actual_rate)

    return ece


def compute_binary_cross_entropy(
    probs: np.ndarray,
    errors: np.ndarray
) -> float:
    """
    Compute binary cross-entropy for error prediction.

    Args:
        probs: (N,) array of predicted error probabilities
        errors: (N,) array of binary error indicators (1=error, 0=correct)

    Returns:
        Binary cross-entropy (lower is better)
    """
    # Clip to avoid log(0)
    eps = 1e-10
    probs_clipped = np.clip(probs, eps, 1 - eps)

    bce = -np.mean(
        errors * np.log(probs_clipped) +
        (1 - errors) * np.log(1 - probs_clipped)
    )

    return bce


def compute_error_prediction_metrics(
    scores: np.ndarray,
    calibrated_probs: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10
) -> ErrorPredictionMetrics:
    """
    Compute all error prediction metrics.

    Args:
        scores: (N,) array of raw uncertainty scores (for ROCAUC)
        calibrated_probs: (N,) array of calibrated error probabilities (for ECE, BCE)
        errors: (N,) array of binary error indicators
        n_bins: Number of bins for ECE

    Returns:
        ErrorPredictionMetrics object
    """
    return ErrorPredictionMetrics(
        rocauc=compute_rocauc(scores, errors),
        ece=compute_ece(calibrated_probs, errors, n_bins),
        binary_cross_entropy=compute_binary_cross_entropy(calibrated_probs, errors),
    )


def compare_calibration(
    scores: np.ndarray,
    errors: np.ndarray,
    calibrated_probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics before and after calibration.

    Args:
        scores: (N,) array of raw uncertainty scores
        errors: (N,) array of binary error indicators
        calibrated_probs: (N,) array of calibrated error probabilities
        n_bins: Number of bins for ECE

    Returns:
        Dict with 'before' and 'after' metrics
    """
    # Before calibration: use raw scores as probabilities
    # (clip to [0, 1] for valid probabilities)
    raw_probs = np.clip(scores, 0.0, 1.0)

    before_metrics = ErrorPredictionMetrics(
        rocauc=compute_rocauc(scores, errors),
        ece=compute_ece(raw_probs, errors, n_bins),
        binary_cross_entropy=compute_binary_cross_entropy(raw_probs, errors),
    )

    after_metrics = ErrorPredictionMetrics(
        rocauc=compute_rocauc(scores, errors),  # ROCAUC unchanged by calibration
        ece=compute_ece(calibrated_probs, errors, n_bins),
        binary_cross_entropy=compute_binary_cross_entropy(calibrated_probs, errors),
    )

    return {
        "before": before_metrics.to_dict(),
        "after": after_metrics.to_dict(),
    }
