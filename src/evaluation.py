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
    ece: float = 0.0  # Classification ECE

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "cross_entropy": self.cross_entropy,
            "normalized_cross_entropy": self.normalized_cross_entropy,
            "ece": self.ece,
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


def compute_classification_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for multiclass classification.
    Uses equal-width binning.

    ECE measures the difference between confidence (max probability) and accuracy.
    Samples are binned by confidence, and ECE is the weighted average of
    |accuracy - confidence| in each bin.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        n_bins: Number of bins

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    n_samples = len(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_count = in_bin.sum()

        if bin_count > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += (bin_count / n_samples) * abs(avg_accuracy - avg_confidence)

    return ece


def compute_classification_ece_uniform_mass(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = None,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for multiclass classification.
    Uses uniform-mass (equal-frequency) binning with Scott's rule.

    This is equivalent to computing error detection ECE on max_proba_complement,
    ensuring consistency between classification and error detection metrics.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        n_bins: Number of bins. If None, uses Scott's rule: B = floor(2 * N^(1/3))

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    # Convert to error detection format: score = 1 - confidence, error = 1 - correct
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    errors = (predictions != labels).astype(float)
    scores = 1 - confidences  # max_proba_complement

    n_samples = len(labels)

    # Scott's rule for number of bins
    if n_bins is None:
        n_bins = int(2 * (n_samples ** (1/3)))

    # Compute quantile boundaries for uniform mass bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(scores, quantiles)

    # Make boundaries unique (handle ties)
    bin_boundaries = np.unique(bin_boundaries)
    actual_n_bins = len(bin_boundaries) - 1

    if actual_n_bins == 0:
        # All scores are identical
        return abs(scores[0] - errors.mean())

    ece = 0.0
    for i in range(actual_n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == actual_n_bins - 1:
            # Include right edge for last bin
            in_bin = (scores >= bin_lower) & (scores <= bin_upper)
        else:
            in_bin = (scores >= bin_lower) & (scores < bin_upper)

        bin_count = in_bin.sum()

        if bin_count > 0:
            # Average score in bin (= 1 - avg_confidence)
            avg_score = scores[in_bin].mean()
            # Actual error rate in bin (= 1 - accuracy)
            actual_rate = errors[in_bin].mean()
            # Weighted absolute difference
            ece += (bin_count / n_samples) * abs(avg_score - actual_rate)

    return ece


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    prior: Optional[np.ndarray] = None,
    n_bins: int = None,
    use_uniform_mass_ece: bool = True,
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        prior: Prior distribution for normalized CE (default: empirical)
        n_bins: Number of bins for ECE (if None, uses Scott's rule for uniform-mass)
        use_uniform_mass_ece: If True, use uniform-mass binning for ECE (default)

    Returns:
        EvaluationMetrics object
    """
    if use_uniform_mass_ece:
        ece = compute_classification_ece_uniform_mass(probs, labels, n_bins)
    else:
        ece = compute_classification_ece(probs, labels, n_bins if n_bins else 10)

    return EvaluationMetrics(
        accuracy=compute_accuracy(probs, labels),
        error_rate=compute_error_rate(probs, labels),
        cross_entropy=compute_cross_entropy(probs, labels),
        normalized_cross_entropy=compute_normalized_cross_entropy(probs, labels, prior),
        ece=ece,
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


def compute_rcc_auc(
    scores: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Compute Risk-Coverage Curve AUC (lower is better).

    Adapted from Vazhentsev et al. (ACL 2022), original implementation:
    github.com/AIRI-Institute/uncertainty_transformers/src/analyze_results.py

    The RCC plots the cumulative average risk (error rate) as a function of
    coverage when samples are ordered from most confident to least confident.

    Args:
        scores: (N,) array of uncertainty scores (higher = more uncertain)
        errors: (N,) array of binary error indicators (1=error, 0=correct)

    Returns:
        RCC-AUC value (lower is better)
    """
    n = len(scores)
    # Authors use (confidence, risk) pairs sorted by confidence descending.
    # Since scores = uncertainty (higher = less confident), use -scores as conf.
    conf = -scores
    risk = errors

    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = [cr_pair[0][1]]
    for i in range(1, n):
        cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])

    auc = 0
    for k in range(n):
        auc += cumulative_risk[k] / (1 + k)

    return auc


def compute_ece(
    probs: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width bins.

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


def compute_ece_uniform_mass(
    scores: np.ndarray,
    errors: np.ndarray,
    n_bins: int = None
) -> float:
    """
    Compute Expected Calibration Error (ECE) using uniform mass (equal-frequency) bins.

    This is appropriate for continuous uncertainty scores that need to be discretized.
    Uses quantile-based binning so each bin has approximately the same number of samples.

    Args:
        scores: (N,) array of uncertainty scores (continuous, will be discretized)
        errors: (N,) array of binary error indicators (1=error, 0=correct)
        n_bins: Number of bins. If None, uses Scott's rule: B = floor(2 * N^(1/3))

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    n_samples = len(errors)

    # Scott's rule for number of bins
    if n_bins is None:
        n_bins = int(2 * (n_samples ** (1/3)))

    # Compute quantile boundaries for uniform mass bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(scores, quantiles)

    # Make boundaries unique (handle ties)
    bin_boundaries = np.unique(bin_boundaries)
    actual_n_bins = len(bin_boundaries) - 1

    if actual_n_bins == 0:
        # All scores are identical
        return abs(scores[0] - errors.mean())

    ece = 0.0
    for i in range(actual_n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == actual_n_bins - 1:
            # Include right edge for last bin
            in_bin = (scores >= bin_lower) & (scores <= bin_upper)
        else:
            in_bin = (scores >= bin_lower) & (scores < bin_upper)

        bin_count = in_bin.sum()

        if bin_count > 0:
            # Average score in bin (treated as predicted probability)
            avg_score = scores[in_bin].mean()
            # Actual error rate in bin
            actual_rate = errors[in_bin].mean()
            # Weighted absolute difference
            ece += (bin_count / n_samples) * abs(avg_score - actual_rate)

    return ece


def compute_ece_discrete(
    calibrated_probs: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Compute ECE for discrete calibrated probabilities (e.g., from Uniform Mass).

    After binning-based calibration, each sample is assigned one of a finite set
    of probability values. Re-discretizing these values would be incorrect.
    Instead, group samples by their assigned probability and compute ECE directly.

    Args:
        calibrated_probs: (N,) array of discrete calibrated error probabilities
        errors: (N,) array of binary error indicators (1=error, 0=correct)

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    n_samples = len(errors)
    unique_probs = np.unique(calibrated_probs)

    ece = 0.0
    for p in unique_probs:
        mask = calibrated_probs == p
        n_bin = mask.sum()
        actual_rate = errors[mask].mean()
        ece += (n_bin / n_samples) * abs(p - actual_rate)

    return ece


def compute_mce_uniform_mass(
    scores: np.ndarray,
    errors: np.ndarray,
    n_bins: int = None
) -> float:
    """
    Compute Maximum Calibration Error (MCE) using uniform mass (equal-frequency) bins.

    Same binning as compute_ece_uniform_mass, but returns the maximum
    |avg_score - actual_error_rate| across bins instead of the weighted average.

    Args:
        scores: (N,) array of uncertainty scores (continuous, will be discretized)
        errors: (N,) array of binary error indicators (1=error, 0=correct)
        n_bins: Number of bins. If None, uses Scott's rule: B = floor(2 * N^(1/3))

    Returns:
        MCE value (lower is better, 0 = perfectly calibrated)
    """
    n_samples = len(errors)

    if n_bins is None:
        n_bins = int(2 * (n_samples ** (1/3)))

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(scores, quantiles)

    bin_boundaries = np.unique(bin_boundaries)
    actual_n_bins = len(bin_boundaries) - 1

    if actual_n_bins == 0:
        return abs(scores[0] - errors.mean())

    max_ce = 0.0
    for i in range(actual_n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == actual_n_bins - 1:
            in_bin = (scores >= bin_lower) & (scores <= bin_upper)
        else:
            in_bin = (scores >= bin_lower) & (scores < bin_upper)

        bin_count = in_bin.sum()

        if bin_count > 0:
            avg_score = scores[in_bin].mean()
            actual_rate = errors[in_bin].mean()
            max_ce = max(max_ce, abs(avg_score - actual_rate))

    return max_ce


def compute_mce_discrete(
    calibrated_probs: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Compute Maximum Calibration Error (MCE) for discrete calibrated probabilities.

    Same grouping as compute_ece_discrete, but returns the maximum
    |p - actual_error_rate| across groups instead of the weighted average.

    Args:
        calibrated_probs: (N,) array of discrete calibrated error probabilities
        errors: (N,) array of binary error indicators (1=error, 0=correct)

    Returns:
        MCE value (lower is better, 0 = perfectly calibrated)
    """
    unique_probs = np.unique(calibrated_probs)

    max_ce = 0.0
    for p in unique_probs:
        mask = calibrated_probs == p
        actual_rate = errors[mask].mean()
        max_ce = max(max_ce, abs(p - actual_rate))

    return max_ce


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
    n_bins: int = 10,
    calibrated_discrete: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics before and after calibration.

    For ECE computation:
    - Before: Use uniform mass (equal-frequency) binning with Scott's rule
      to discretize continuous uncertainty scores
    - After: If calibrated_discrete=True (e.g., Uniform Mass calibration),
      use compute_ece_discrete (no re-binning). Otherwise, use uniform mass
      binning on the continuous calibrated probabilities.

    Args:
        scores: (N,) array of raw uncertainty scores
        errors: (N,) array of binary error indicators
        calibrated_probs: (N,) array of calibrated error probabilities
        n_bins: Number of bins for ECE (after calibration, ignored if discrete)
        calibrated_discrete: If True, calibrated probs are already discrete
            (e.g., from Uniform Mass binning) and should not be re-binned.

    Returns:
        Dict with 'before' and 'after' metrics
    """
    # Before calibration: use raw scores as probabilities
    # (clip to [0, 1] for valid probabilities)
    raw_probs = np.clip(scores, 0.0, 1.0)

    # For ECE before calibration: use uniform mass binning with Scott's rule
    # This properly handles continuous uncertainty scores
    n_bins_scott = int(2 * (len(scores) ** (1/3)))

    # ROCAUC before: computed on original scores
    rocauc_before = compute_rocauc(scores, errors)

    # ROCAUC after: computed on calibrated probabilities
    # For strictly monotonic transforms (PHC-DP/TS/BO), this should be unchanged
    # For Uniform Mass (binning), this may change due to ties
    rocauc_after = compute_rocauc(calibrated_probs, errors)

    before_metrics = ErrorPredictionMetrics(
        rocauc=rocauc_before,
        ece=compute_ece_uniform_mass(raw_probs, errors, n_bins=n_bins_scott),
        binary_cross_entropy=compute_binary_cross_entropy(raw_probs, errors),
    )

    # After calibration: discrete scores skip re-binning
    if calibrated_discrete:
        ece_after = compute_ece_discrete(calibrated_probs, errors)
    else:
        ece_after = compute_ece_uniform_mass(
            calibrated_probs, errors, n_bins=n_bins_scott
        )

    after_metrics = ErrorPredictionMetrics(
        rocauc=rocauc_after,
        ece=ece_after,
        binary_cross_entropy=compute_binary_cross_entropy(calibrated_probs, errors),
    )

    return {
        "before": before_metrics.to_dict(),
        "after": after_metrics.to_dict(),
    }
