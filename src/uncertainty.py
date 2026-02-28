"""
Uncertainty score computation for error prediction.

Given class probabilities from an LLM, compute various uncertainty scores
that can be used to predict whether the model's prediction is likely to be wrong.
"""

import numpy as np
from scipy.special import logsumexp
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class UncertaintyScores:
    """Container for uncertainty scores."""
    max_proba_complement: np.ndarray  # 1 - max(p)
    margin: np.ndarray                 # max(p) - second_max(p)
    doctor: np.ndarray                 # 1 - Gini(p)
    doctor_normalized: np.ndarray      # doctor normalized by max Gini
    predictive_entropy: np.ndarray = field(default=None)  # -Σ p_k log(p_k)
    energy: Optional[np.ndarray] = field(default=None)    # -log Σ exp(l_k), requires logits

    def to_dict(self) -> Dict[str, np.ndarray]:
        d = {
            "max_proba_complement": self.max_proba_complement,
            "margin": self.margin,
            "doctor": self.doctor,
            "doctor_normalized": self.doctor_normalized,
            "predictive_entropy": self.predictive_entropy,
        }
        if self.energy is not None:
            d["energy"] = self.energy
        return d


def compute_max_proba_complement(probs: np.ndarray) -> np.ndarray:
    """
    Compute 1 - max(p) uncertainty score.

    Higher values indicate higher uncertainty (low confidence).

    Args:
        probs: (N, K) array of class probabilities

    Returns:
        (N,) array of uncertainty scores
    """
    return 1.0 - np.max(probs, axis=1)


def compute_margin(probs: np.ndarray) -> np.ndarray:
    """
    Compute margin uncertainty score: max(p) - second_max(p).

    Lower margin indicates higher uncertainty (close competition between classes).
    We return 1 - margin so that higher values = higher uncertainty.

    Args:
        probs: (N, K) array of class probabilities

    Returns:
        (N,) array of uncertainty scores (1 - margin, so higher = more uncertain)
    """
    # Sort probabilities in descending order along class axis
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]

    # Margin = max - second_max
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    # Return 1 - margin so higher values = higher uncertainty
    return 1.0 - margin


def compute_gini(probs: np.ndarray) -> np.ndarray:
    """
    Compute Gini impurity index.

    Gini(p) = 1 - sum(p_k^2)

    For a K-class problem:
    - Gini = 0 when one class has probability 1 (pure)
    - Gini_max = 1 - 1/K when uniform distribution (most impure)

    Args:
        probs: (N, K) array of class probabilities

    Returns:
        (N,) array of Gini indices
    """
    return 1.0 - np.sum(probs ** 2, axis=1)


def compute_doctor(probs: np.ndarray) -> np.ndarray:
    """
    Compute Doctor uncertainty score: 1 - Gini(p).

    Doctor = 1 - Gini = sum(p_k^2)

    Higher doctor = lower Gini = more concentrated (more confident).
    We return 1 - doctor so that higher values = higher uncertainty.

    Actually, we want higher score = higher uncertainty, so we return Gini directly.

    Args:
        probs: (N, K) array of class probabilities

    Returns:
        (N,) array of uncertainty scores (Gini index)
    """
    # Gini index: higher = more uncertain
    return compute_gini(probs)


def compute_doctor_normalized(probs: np.ndarray) -> np.ndarray:
    """
    Compute normalized Doctor uncertainty score.

    Normalized by max Gini (uniform distribution): Gini_max = 1 - 1/K

    Result is in [0, 1] where:
    - 0 = one class has probability 1 (most confident)
    - 1 = uniform distribution (most uncertain)

    Args:
        probs: (N, K) array of class probabilities

    Returns:
        (N,) array of normalized uncertainty scores
    """
    n_classes = probs.shape[1]
    gini_max = 1.0 - 1.0 / n_classes

    gini = compute_gini(probs)

    return gini / gini_max


def compute_predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Compute predictive entropy: -Σ p_k log(p_k).

    Higher values indicate higher uncertainty.
    Bounded in [0, log(K)] where K is the number of classes.

    Args:
        probs: (N, K) array of class probabilities

    Returns:
        (N,) array of entropy values
    """
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs_clipped), axis=1)


def compute_energy(logits: np.ndarray) -> np.ndarray:
    """
    Compute energy score: -log Σ exp(l_k).

    Higher values (closer to 0) indicate higher uncertainty.
    Range: (-∞, 0).

    Reference: Liu et al. (2020) "Energy-based Out-of-distribution Detection".

    Args:
        logits: (N, K) array of raw logits (pre-softmax)

    Returns:
        (N,) array of energy scores
    """
    return -logsumexp(logits, axis=1)


def compute_uncertainty_scores(
    probs: np.ndarray,
    logits: Optional[np.ndarray] = None,
) -> UncertaintyScores:
    """
    Compute all uncertainty scores from class probabilities and optionally logits.

    Args:
        probs: (N, K) array of class probabilities
        logits: (N, K) array of raw logits, optional. Needed for energy score.

    Returns:
        UncertaintyScores object with all scores
    """
    energy = compute_energy(logits) if logits is not None else None

    return UncertaintyScores(
        max_proba_complement=compute_max_proba_complement(probs),
        margin=compute_margin(probs),
        doctor=compute_doctor(probs),
        doctor_normalized=compute_doctor_normalized(probs),
        predictive_entropy=compute_predictive_entropy(probs),
        energy=energy,
    )


def get_predictions_and_errors(
    probs: np.ndarray,
    labels: np.ndarray
) -> tuple:
    """
    Get predictions and error indicators from probabilities and labels.

    Args:
        probs: (N, K) array of class probabilities
        labels: (N,) array of true labels

    Returns:
        predictions: (N,) array of predicted class indices
        errors: (N,) array of binary error indicators (1 if wrong, 0 if correct)
    """
    predictions = np.argmax(probs, axis=1)
    errors = (predictions != labels).astype(int)
    return predictions, errors


def prepare_calibration_data(
    probs: np.ndarray,
    labels: np.ndarray,
    score_name: str = "max_proba_complement",
    logits: Optional[np.ndarray] = None,
) -> tuple:
    """
    Prepare data for uncertainty score calibration.

    Args:
        probs: (N, K) array of class probabilities
        labels: (N,) array of true labels
        score_name: Which uncertainty score to use
        logits: (N, K) array of raw logits, optional. Needed for energy score.

    Returns:
        scores: (N,) array of uncertainty scores
        errors: (N,) array of binary error indicators
    """
    scores_obj = compute_uncertainty_scores(probs, logits=logits)
    scores_dict = scores_obj.to_dict()

    if score_name not in scores_dict:
        raise ValueError(f"Unknown score name: {score_name}. "
                        f"Available: {list(scores_dict.keys())}")

    scores = scores_dict[score_name]
    _, errors = get_predictions_and_errors(probs, labels)

    return scores, errors
