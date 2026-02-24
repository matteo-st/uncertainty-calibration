"""
Mahalanobis distance uncertainty score computation.

Implements the Mahalanobis distance (MD) method from:
    Vazhentsev et al. (ACL 2022)
    "Uncertainty Estimation of Transformer Predictions for Misclassification Detection"

The MD score measures the distance between a test instance's hidden
representation and the closest class-conditional Gaussian fitted on
training data:

    u_MD(x) = min_c (h - mu_c)^T Sigma^{-1} (h - mu_c)

Higher MD score = higher uncertainty (farther from any class centroid).

Key property: MD scores are unbounded positive reals [0, +inf).
They are NOT interpretable as probabilities until calibrated.
"""

import numpy as np
from typing import Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_centroids(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute class-conditional centroids (means).

    Args:
        features: (N, D) array of hidden representations
        labels: (N,) array of class labels (integers)
        num_classes: Number of classes (inferred from labels if None)

    Returns:
        (C, D) array of class centroids, where C = num_classes
    """
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    D = features.shape[1]
    centroids = np.zeros((num_classes, D))

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            logger.warning(f"No samples for class {c}. Centroid set to zeros.")
            continue
        centroids[c] = features[mask].mean(axis=0)
        logger.debug(f"Class {c}: {mask.sum()} samples, "
                     f"centroid norm = {np.linalg.norm(centroids[c]):.4f}")

    return centroids


def compute_covariance_inverse(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Compute shared covariance matrix and its inverse.

    The shared covariance is computed as:
        Sigma = (1/N) * sum_c sum_{x in class c} (x - mu_c)(x - mu_c)^T

    where mu_c is the centroid of class c. This pools covariances across
    classes, assuming a shared covariance structure.

    If the covariance matrix is singular (which can happen when D > N),
    the pseudo-inverse is used instead, following the reference implementation.

    Args:
        features: (N, D) array of hidden representations
        labels: (N,) array of class labels
        centroids: (C, D) array of class centroids

    Returns:
        (D, D) inverse (or pseudo-inverse) of the shared covariance matrix
    """
    N, D = features.shape
    num_classes = centroids.shape[0]

    # Center features by their class centroid
    centered = np.zeros_like(features)
    for c in range(num_classes):
        mask = labels == c
        centered[mask] = features[mask] - centroids[c]

    # Shared covariance: (1/N) * X_centered^T @ X_centered
    covariance = (centered.T @ centered) / N

    logger.info(f"Covariance matrix shape: {covariance.shape}")
    logger.info(f"Covariance matrix rank: {np.linalg.matrix_rank(covariance)}")

    # Compute inverse (or pseudo-inverse if singular)
    try:
        cov_inv = np.linalg.inv(covariance)
        logger.info("Used standard matrix inverse")
    except np.linalg.LinAlgError:
        logger.warning(
            "Covariance matrix is singular. Using pseudo-inverse. "
            f"Matrix rank {np.linalg.matrix_rank(covariance)} < dimension {D}."
        )
        cov_inv = np.linalg.pinv(covariance)

    return cov_inv


def mahalanobis_distance(
    features: np.ndarray,
    centroids: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distance for each sample to the closest class centroid.

        u_MD(x) = min_c (h - mu_c)^T Sigma^{-1} (h - mu_c)

    Higher values indicate higher uncertainty (farther from training distribution).

    Args:
        features: (N, D) array of hidden representations
        centroids: (C, D) array of class centroids
        cov_inv: (D, D) inverse covariance matrix

    Returns:
        (N,) array of MD scores (minimum distance over classes)
    """
    N = features.shape[0]
    C = centroids.shape[0]

    # Compute distance to each class centroid
    # For each class c: d_c(x) = (x - mu_c)^T Sigma^{-1} (x - mu_c)
    distances = np.zeros((N, C))

    for c in range(C):
        diff = features - centroids[c]  # (N, D)
        # (N, D) @ (D, D) -> (N, D), then element-wise multiply and sum
        distances[:, c] = np.sum(diff @ cov_inv * diff, axis=1)

    # Take minimum distance over classes
    md_scores = distances.min(axis=1)

    logger.info(f"MD scores: min={md_scores.min():.4f}, "
                f"max={md_scores.max():.4f}, "
                f"mean={md_scores.mean():.4f}, "
                f"median={np.median(md_scores):.4f}")

    return md_scores


class MahalanobisScorer:
    """
    Stateful Mahalanobis distance scorer.

    Fit on training data (compute centroids and covariance inverse),
    then score new samples.

    Usage:
        scorer = MahalanobisScorer()
        scorer.fit(train_features, train_labels)
        scores = scorer.score(test_features)
    """

    def __init__(self, num_classes: Optional[int] = None):
        """
        Initialize the scorer.

        Args:
            num_classes: Number of classes (inferred from labels if None)
        """
        self.num_classes = num_classes
        self.centroids = None
        self.cov_inv = None
        self._fitted = False

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
    ) -> "MahalanobisScorer":
        """
        Fit the scorer on training data.

        Computes class-conditional centroids and the shared covariance
        matrix inverse from the training representations.

        Args:
            train_features: (N_train, D) training hidden representations
            train_labels: (N_train,) training labels

        Returns:
            self (for method chaining)
        """
        logger.info(f"Fitting MahalanobisScorer on {len(train_features)} samples, "
                    f"dim={train_features.shape[1]}")

        self.centroids = compute_centroids(
            train_features, train_labels, self.num_classes
        )
        self.cov_inv = compute_covariance_inverse(
            train_features, train_labels, self.centroids
        )
        self._fitted = True

        logger.info(f"MahalanobisScorer fitted: {self.centroids.shape[0]} classes, "
                    f"dim={self.centroids.shape[1]}")

        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute MD scores for new samples.

        Args:
            features: (N, D) hidden representations to score

        Returns:
            (N,) array of MD scores (higher = more uncertain)
        """
        if not self._fitted:
            raise RuntimeError("MahalanobisScorer not fitted. Call fit() first.")

        return mahalanobis_distance(features, self.centroids, self.cov_inv)

    def save(self, path: str) -> None:
        """
        Save the fitted scorer to a pickle file.

        Args:
            path: File path (.pkl)
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted scorer.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "centroids": self.centroids,
            "cov_inv": self.cov_inv,
            "num_classes": self.num_classes,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"MahalanobisScorer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MahalanobisScorer":
        """
        Load a fitted scorer from a pickle file.

        Args:
            path: File path (.pkl)

        Returns:
            Fitted MahalanobisScorer instance
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        scorer = cls(num_classes=state["num_classes"])
        scorer.centroids = state["centroids"]
        scorer.cov_inv = state["cov_inv"]
        scorer._fitted = True

        logger.info(f"MahalanobisScorer loaded from {path}: "
                    f"{scorer.centroids.shape[0]} classes, "
                    f"dim={scorer.centroids.shape[1]}")

        return scorer

    def get_stats(self) -> dict:
        """Return summary statistics of the fitted scorer."""
        if not self._fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "num_classes": self.centroids.shape[0],
            "hidden_dim": self.centroids.shape[1],
            "centroid_norms": [
                float(np.linalg.norm(self.centroids[c]))
                for c in range(self.centroids.shape[0])
            ],
            "cov_inv_norm": float(np.linalg.norm(self.cov_inv)),
        }
