"""
Calibration methods for LLM posteriors.

Implements the calibration techniques from:
"Unsupervised Calibration through Prior Adaptation for Text Classification
using Large Language Models" (Estienne et al., 2023)
"""

import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod


class Calibrator(ABC):
    """Base class for calibration methods."""

    @abstractmethod
    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        """
        Fit the calibrator on training data.

        Args:
            probs_train: Uncalibrated probabilities, shape (n_samples, n_classes)
            labels_train: True labels (optional, for supervised methods)
        """
        pass

    @abstractmethod
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            probs: Uncalibrated probabilities, shape (n_samples, n_classes)

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes)
        """
        pass


class NoCalibration(Calibrator):
    """Baseline: no calibration applied."""

    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        pass

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        return probs


class ContentFreeCalibrator(Calibrator):
    """
    Content-free calibration (Zhao et al., 2021).

    Uses content-free inputs (like "[MASK]", "N/A", "") to estimate the
    model's prior P(y|e), then applies prior correction.
    """

    def __init__(self, content_free_probs: np.ndarray, target_prior: Optional[np.ndarray] = None):
        """
        Args:
            content_free_probs: Probabilities from content-free inputs,
                               shape (n_inputs, n_classes) or (n_classes,)
            target_prior: Target prior distribution (default: uniform)
        """
        # Average if multiple content-free inputs
        if content_free_probs.ndim == 2:
            self.model_prior = content_free_probs.mean(axis=0)
        else:
            self.model_prior = content_free_probs

        self.n_classes = len(self.model_prior)
        self.target_prior = target_prior if target_prior is not None else np.ones(self.n_classes) / self.n_classes

    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        # Content-free calibration doesn't use training data
        pass

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply prior correction (Eq. 5 in paper).

        P_hat(y|q,e) = delta * P(y|q,e) * [P_hat(y|e) / P(y|e)]
        """
        # Compute correction factor
        correction = self.target_prior / (self.model_prior + 1e-10)

        # Apply correction
        calibrated = probs * correction

        # Renormalize (delta factor)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated


class UCPANaiveCalibrator(Calibrator):
    """
    UCPA-Naive: Unsupervised Calibration through Prior Adaptation (naive version).

    Estimates P(y|e) by averaging posteriors over unlabeled training data (Eq. 6),
    assumes uniform target prior.
    """

    def __init__(self, target_prior: Optional[np.ndarray] = None):
        """
        Args:
            target_prior: Target prior (default: uniform)
        """
        self.target_prior = target_prior
        self.model_prior = None

    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        """
        Estimate model's prior from training posteriors (Eq. 6).

        P(y|e) ≈ (1/N) * sum_i P(y|q^(i), e)
        """
        self.model_prior = probs_train.mean(axis=0)
        self.n_classes = probs_train.shape[1]

        if self.target_prior is None:
            self.target_prior = np.ones(self.n_classes) / self.n_classes

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply prior correction (Eq. 5)."""
        if self.model_prior is None:
            raise RuntimeError("Calibrator must be fit before calibrating")

        correction = self.target_prior / (self.model_prior + 1e-10)
        calibrated = probs * correction
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated


class SUCPANaiveCalibrator(UCPANaiveCalibrator):
    """
    SUCPA-Naive: Semi-Unsupervised Calibration through Prior Adaptation.

    Same as UCPA-Naive but uses estimated class priors from training labels
    instead of assuming uniform prior.
    """

    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        """
        Estimate model's prior and target prior from training data.
        """
        self.model_prior = probs_train.mean(axis=0)
        self.n_classes = probs_train.shape[1]

        if labels_train is not None:
            # Estimate target prior from label frequencies
            label_counts = np.bincount(labels_train, minlength=self.n_classes)
            self.target_prior = label_counts / len(labels_train)
        else:
            self.target_prior = np.ones(self.n_classes) / self.n_classes


class UCPACalibrator(Calibrator):
    """
    UCPA: Unsupervised Calibration through Prior Adaptation (iterative version).

    Uses iterative estimation of beta parameters (Eq. 10) which is equivalent
    to logistic regression with alpha=1.
    """

    def __init__(
        self,
        target_prior: Optional[np.ndarray] = None,
        n_iterations: int = 10,
        tol: float = 1e-6,
    ):
        """
        Args:
            target_prior: Target prior (default: uniform)
            n_iterations: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.target_prior = target_prior
        self.n_iterations = n_iterations
        self.tol = tol
        self.beta = None

    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        """
        Iteratively estimate beta parameters (Eq. 10).

        beta_k = log(N_k/N) - log[(1/N) * sum_i P(y_k|q^(i),e) * e^{gamma^(i)}]

        where gamma^(i) = -log[sum_k' P(y_k'|q^(i),e) * e^{beta_k'}]
        """
        n_samples, n_classes = probs_train.shape
        self.n_classes = n_classes

        if self.target_prior is None:
            self.target_prior = np.ones(n_classes) / n_classes

        # Initialize beta to zeros (gamma^(i) = 0 initially)
        beta = np.zeros(n_classes)

        for iteration in range(self.n_iterations):
            beta_old = beta.copy()

            # Compute gamma^(i) for each sample (Eq. 8)
            # gamma^(i) = -log[sum_k P(y_k|q^(i),e) * e^{beta_k}]
            weighted = probs_train * np.exp(beta)  # (n_samples, n_classes)
            gamma = -np.log(weighted.sum(axis=1) + 1e-10)  # (n_samples,)

            # Update beta (Eq. 10)
            # beta_k = log(prior_k) - log[(1/N) * sum_i P(y_k|q^(i),e) * e^{gamma^(i)}]
            exp_gamma = np.exp(gamma)[:, np.newaxis]  # (n_samples, 1)
            weighted_mean = (probs_train * exp_gamma).mean(axis=0)  # (n_classes,)
            beta = np.log(self.target_prior + 1e-10) - np.log(weighted_mean + 1e-10)

            # Check convergence
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        self.beta = beta

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration using learned beta (Eq. 7 with alpha=1).

        log P_tilde(y_k|q,e) = gamma + log P(y_k|q,e) + beta_k
        """
        if self.beta is None:
            raise RuntimeError("Calibrator must be fit before calibrating")

        # Apply transformation in log space
        log_probs = np.log(probs + 1e-10)
        log_calibrated = log_probs + self.beta

        # Normalize (this computes gamma implicitly)
        calibrated = np.exp(log_calibrated)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated


class SUCPACalibrator(UCPACalibrator):
    """
    SUCPA: Semi-Unsupervised Calibration through Prior Adaptation.

    Same as UCPA but uses estimated class priors from training labels.
    """

    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        """Estimate target prior from labels, then fit iteratively."""
        n_classes = probs_train.shape[1]

        if labels_train is not None:
            label_counts = np.bincount(labels_train, minlength=n_classes)
            self.target_prior = label_counts / len(labels_train)
        else:
            self.target_prior = np.ones(n_classes) / n_classes

        # Call parent fit with the estimated prior
        super().fit(probs_train, labels_train)


class AffineCalibrator(Calibrator):
    """
    Supervised affine calibration (Eq. 7).

    Learns both alpha (temperature-like) and beta (bias) parameters
    by minimizing cross-entropy on labeled data.
    """

    def __init__(
        self,
        learn_alpha: bool = True,
        n_iterations: int = 100,
        lr: float = 0.1,
    ):
        """
        Args:
            learn_alpha: Whether to learn alpha (if False, fix alpha=1)
            n_iterations: Number of optimization iterations
            lr: Learning rate
        """
        self.learn_alpha = learn_alpha
        self.n_iterations = n_iterations
        self.lr = lr
        self.alpha = None
        self.beta = None

    def fit(self, probs_train: np.ndarray, labels_train: Optional[np.ndarray] = None):
        """
        Minimize cross-entropy loss (Eq. 9) to learn parameters.
        """
        if labels_train is None:
            raise ValueError("AffineCalibrator requires labeled training data")

        n_samples, n_classes = probs_train.shape

        # Initialize parameters
        alpha = 1.0
        beta = np.zeros(n_classes)

        log_probs = np.log(probs_train + 1e-10)

        for _ in range(self.n_iterations):
            # Forward pass
            if self.learn_alpha:
                logits = alpha * log_probs + beta
            else:
                logits = log_probs + beta

            # Softmax
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Gradient of cross-entropy
            # d_loss/d_logits = calibrated - one_hot(labels)
            one_hot = np.zeros_like(calibrated)
            one_hot[np.arange(n_samples), labels_train] = 1
            grad_logits = (calibrated - one_hot) / n_samples

            # Update beta
            grad_beta = grad_logits.sum(axis=0)
            beta -= self.lr * grad_beta

            # Update alpha
            if self.learn_alpha:
                grad_alpha = (grad_logits * log_probs).sum()
                alpha -= self.lr * grad_alpha
                alpha = max(0.01, alpha)  # Keep alpha positive

        self.alpha = alpha
        self.beta = beta

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned affine transformation."""
        if self.beta is None:
            raise RuntimeError("Calibrator must be fit before calibrating")

        log_probs = np.log(probs + 1e-10)

        if self.learn_alpha:
            logits = self.alpha * log_probs + self.beta
        else:
            logits = log_probs + self.beta

        # Softmax
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return calibrated
