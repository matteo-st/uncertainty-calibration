"""
Uncertainty score calibration methods for error prediction.

Methods:
- PHC-DP (Direction Preserving / Platt Scaling): sigmoid(alpha * score + beta)
- PHC-TS (Temperature Scaling): sigmoid(alpha * score), beta=0
- PHC-BO (Bias Only): sigmoid(score + beta), alpha=1
- Uniform Mass Calibration: Quantile binning with empirical error rates

Reference: Estienne et al. (2026) "Adapting Language Models to Produce
Good Class Probabilities for Classification Tasks" (TMLR)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function


class ScoreCalibrator(ABC):
    """Abstract base class for uncertainty score calibrators."""

    @abstractmethod
    def fit(self, scores: np.ndarray, errors: np.ndarray) -> None:
        """
        Fit the calibrator on training data.

        Args:
            scores: (N,) array of uncertainty scores
            errors: (N,) array of binary error indicators (1=error, 0=correct)
        """
        pass

    @abstractmethod
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform uncertainty scores to calibrated error probabilities.

        Args:
            scores: (N,) array of uncertainty scores

        Returns:
            (N,) array of calibrated error probabilities
        """
        pass

    def fit_calibrate(
        self,
        train_scores: np.ndarray,
        train_errors: np.ndarray,
        test_scores: np.ndarray
    ) -> np.ndarray:
        """Fit on training data and calibrate test scores."""
        self.fit(train_scores, train_errors)
        return self.calibrate(test_scores)


class NoCalibration(ScoreCalibrator):
    """
    No calibration baseline.

    Simply returns the raw uncertainty scores as error probabilities.
    This assumes the scores are already in [0, 1] range.
    """

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> None:
        """No fitting needed."""
        pass

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Return scores as-is (clipped to [0, 1])."""
        return np.clip(scores, 0.0, 1.0)


class PlattScaling(ScoreCalibrator):
    """
    Platt Scaling / Direction Preserving (PHC-DP) calibration.

    First transforms scores to logits, then applies affine transformation:
        P(error | score) = sigmoid(alpha * logit(score) + beta)

    where logit(x) = log(x / (1-x))

    The logit transform maps [0,1] scores to (-inf, +inf), which is the natural
    domain for the sigmoid function. This ensures consistent behavior regardless
    of the error rate.

    Parameters alpha and beta are learned by minimizing binary cross-entropy.

    Variants (all use logit transform for consistency):
    - PHC-DP: Learn both alpha and beta (default)
    - PHC-TS: Learn only alpha, beta=0 (temperature scaling)
    - PHC-BO: Learn only beta, alpha=1 (bias only)
    """

    def __init__(
        self,
        learn_alpha: bool = True,
        learn_beta: bool = True,
        n_iterations: int = 1000,
        lr: float = 0.1,
        patience: int = 10,
    ):
        """
        Initialize Platt scaling calibrator.

        Args:
            learn_alpha: If True, learn the scaling parameter alpha
            learn_beta: If True, learn the bias parameter beta
            n_iterations: Maximum number of optimization iterations
            lr: Learning rate
            patience: Early stopping patience (stop if loss doesn't improve for this many steps)
        """
        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        self.n_iterations = n_iterations
        self.lr = lr
        self.patience = patience

        # Initialize parameters
        self.alpha = 1.0
        self.beta = 0.0
        self.loss_history = []
        self.n_iterations_run = 0

    def _to_logits(self, scores: np.ndarray) -> np.ndarray:
        """Transform scores from [0,1] to logits (-inf, +inf)."""
        eps = 1e-7
        scores_clipped = np.clip(scores, eps, 1 - eps)
        return np.log(scores_clipped / (1 - scores_clipped))

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> None:
        """
        Fit alpha and beta by minimizing binary cross-entropy with gradient descent.
        Uses early stopping if loss doesn't decrease for `patience` steps.

        Args:
            scores: (N,) array of uncertainty scores
            errors: (N,) array of binary error indicators
        """
        # Transform scores to logits
        score_logits = self._to_logits(scores)

        # Initialize parameters
        alpha = 1.0
        beta = 0.0

        # Early stopping tracking
        best_loss = float('inf')
        best_alpha = alpha
        best_beta = beta
        steps_without_improvement = 0
        self.loss_history = []

        n_samples = len(scores)

        for iteration in range(self.n_iterations):
            # Forward pass
            logits = alpha * score_logits + beta
            probs = expit(logits)

            # Compute binary cross-entropy loss
            eps = 1e-10
            probs_clipped = np.clip(probs, eps, 1 - eps)
            loss = -np.mean(
                errors * np.log(probs_clipped) + (1 - errors) * np.log(1 - probs_clipped)
            )
            self.loss_history.append(loss)

            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
                best_beta = beta
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= self.patience:
                    break

            # Compute gradients
            # d_loss/d_logits = probs - errors
            grad_logits = (probs - errors) / n_samples

            # Update beta
            if self.learn_beta:
                grad_beta = grad_logits.sum()
                beta -= self.lr * grad_beta

            # Update alpha
            if self.learn_alpha:
                grad_alpha = (grad_logits * score_logits).sum()
                alpha -= self.lr * grad_alpha

        # Use best parameters
        self.alpha = best_alpha
        self.beta = best_beta
        self.n_iterations_run = iteration + 1

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to get calibrated error probabilities.

        Args:
            scores: (N,) array of uncertainty scores

        Returns:
            (N,) array of calibrated error probabilities
        """
        score_logits = self._to_logits(scores)
        logits = self.alpha * score_logits + self.beta
        return expit(logits)

    def get_params(self) -> dict:
        """Return learned parameters."""
        return {"alpha": self.alpha, "beta": self.beta}


class TemperatureScaling(PlattScaling):
    """
    Temperature Scaling (PHC-TS) calibration.

    Special case of Platt scaling with beta=0:
        P(error | score) = sigmoid(alpha * logit(score))

    Uses logit transform (same as PHC-DP) for consistency.
    Only learns the temperature parameter alpha.
    """

    def __init__(self, n_iterations: int = 1000, lr: float = 0.1, patience: int = 10):
        super().__init__(
            learn_alpha=True, learn_beta=False,
            n_iterations=n_iterations, lr=lr, patience=patience
        )


class BiasOnlyCalibration(PlattScaling):
    """
    Bias Only (PHC-BO) calibration.

    Special case of Platt scaling with alpha=1:
        P(error | score) = sigmoid(logit(score) + beta)

    Uses logit transform (same as PHC-DP) for consistency.
    Only learns the bias parameter beta.
    """

    def __init__(self, n_iterations: int = 1000, lr: float = 0.1, patience: int = 10):
        super().__init__(
            learn_alpha=False, learn_beta=True,
            n_iterations=n_iterations, lr=lr, patience=patience
        )


class UniformMassCalibration(ScoreCalibrator):
    """
    Uniform Mass Calibration (Quantile Binning).

    Divides data into bins with equal number of samples (quantile binning),
    then computes empirical error rate in each bin.

    For a new score, finds the appropriate bin and returns the
    empirical error rate of that bin.
    """

    def __init__(self, n_bins: int = None, use_scott_rule: bool = True):
        """
        Initialize uniform mass calibrator.

        Args:
            n_bins: Number of bins (if None and use_scott_rule=True, computed from data)
            use_scott_rule: If True and n_bins is None, use Scott's rule: B = 2 * N^(1/3)
        """
        self.n_bins = n_bins
        self.use_scott_rule = use_scott_rule
        self.n_bins_actual = None
        self.bin_edges = None
        self.bin_error_rates = None

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> None:
        """
        Compute bin edges (quantiles) and empirical error rates.

        Args:
            scores: (N,) array of uncertainty scores
            errors: (N,) array of binary error indicators
        """
        n_samples = len(scores)

        # Determine number of bins
        if self.n_bins is not None:
            self.n_bins_actual = self.n_bins
        elif self.use_scott_rule:
            # Scott's rule: B = 2 * N^(1/3)
            self.n_bins_actual = int(2 * (n_samples ** (1/3)))
        else:
            self.n_bins_actual = 10  # default

        # Compute quantile-based bin edges
        quantiles = np.linspace(0, 100, self.n_bins_actual + 1)
        self.bin_edges = np.percentile(scores, quantiles)

        # Ensure unique bin edges (handle repeated values)
        self.bin_edges = np.unique(self.bin_edges)
        actual_n_bins = len(self.bin_edges) - 1

        # Compute empirical error rate in each bin
        self.bin_error_rates = np.zeros(actual_n_bins)

        for i in range(actual_n_bins):
            if i == actual_n_bins - 1:
                # Last bin includes right edge
                mask = (scores >= self.bin_edges[i]) & (scores <= self.bin_edges[i + 1])
            else:
                mask = (scores >= self.bin_edges[i]) & (scores < self.bin_edges[i + 1])

            if mask.sum() > 0:
                self.bin_error_rates[i] = errors[mask].mean()
            else:
                # Empty bin: use overall error rate
                self.bin_error_rates[i] = errors.mean()

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Map scores to calibrated probabilities via binning.

        Args:
            scores: (N,) array of uncertainty scores

        Returns:
            (N,) array of calibrated error probabilities
        """
        if self.bin_edges is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        # Digitize: find bin index for each score
        # np.digitize returns 1-indexed, and we need 0-indexed
        bin_indices = np.digitize(scores, self.bin_edges[1:-1])

        # Clip to valid range
        bin_indices = np.clip(bin_indices, 0, len(self.bin_error_rates) - 1)

        return self.bin_error_rates[bin_indices]

    def get_params(self) -> dict:
        """Return bin edges and error rates."""
        return {
            "n_bins_actual": self.n_bins_actual,
            "bin_edges": self.bin_edges.tolist() if self.bin_edges is not None else None,
            "bin_error_rates": self.bin_error_rates.tolist() if self.bin_error_rates is not None else None,
        }


class IsotonicCalibration(ScoreCalibrator):
    """
    Isotonic Regression calibration.

    Fits a non-decreasing function from scores to error probabilities
    using isotonic regression.
    """

    def __init__(self):
        self.ir = None

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> None:
        """
        Fit isotonic regression.

        Args:
            scores: (N,) array of uncertainty scores
            errors: (N,) array of binary error indicators
        """
        from sklearn.isotonic import IsotonicRegression

        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.ir.fit(scores, errors)

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            scores: (N,) array of uncertainty scores

        Returns:
            (N,) array of calibrated error probabilities
        """
        if self.ir is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self.ir.predict(scores)


def get_calibrator(method: str, **kwargs) -> ScoreCalibrator:
    """
    Factory function to get a calibrator by name.

    Args:
        method: Calibration method name
            - 'none': No calibration
            - 'phc_dp' or 'platt': Platt scaling (alpha and beta)
            - 'phc_ts' or 'temperature': Temperature scaling (only alpha)
            - 'phc_bo' or 'bias': Bias only (only beta)
            - 'uniform_mass' or 'quantile': Uniform mass binning
            - 'isotonic': Isotonic regression
        **kwargs: Additional arguments for the calibrator

    Returns:
        ScoreCalibrator instance
    """
    method = method.lower()

    if method in ['none', 'no_calibration']:
        return NoCalibration()
    elif method in ['phc_dp', 'platt', 'platt_scaling', 'direction_preserving']:
        return PlattScaling(**kwargs)
    elif method in ['phc_ts', 'temperature', 'temperature_scaling']:
        return TemperatureScaling(**kwargs)
    elif method in ['phc_bo', 'bias', 'bias_only']:
        return BiasOnlyCalibration(**kwargs)
    elif method in ['uniform_mass', 'quantile', 'quantile_binning']:
        return UniformMassCalibration(**kwargs)
    elif method in ['isotonic', 'isotonic_regression']:
        return IsotonicCalibration()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
