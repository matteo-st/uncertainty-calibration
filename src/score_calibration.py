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

    Maps uncertainty score to error probability via logistic function:
        P(error | score) = sigmoid(alpha * score + beta)

    Parameters alpha and beta are learned by minimizing binary cross-entropy.

    Variants:
    - PHC-DP: Learn both alpha and beta (default)
    - PHC-TS: Learn only alpha, beta=0 (temperature scaling)
    - PHC-BO: Learn only beta, alpha=1 (bias only)
    """

    def __init__(
        self,
        learn_alpha: bool = True,
        learn_beta: bool = True,
        regularization: float = 0.0
    ):
        """
        Initialize Platt scaling calibrator.

        Args:
            learn_alpha: If True, learn the scaling parameter alpha
            learn_beta: If True, learn the bias parameter beta
            regularization: L2 regularization strength (default: 0)
        """
        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        self.regularization = regularization

        # Initialize parameters
        self.alpha = 1.0
        self.beta = 0.0

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> None:
        """
        Fit alpha and beta by minimizing binary cross-entropy.

        Args:
            scores: (N,) array of uncertainty scores
            errors: (N,) array of binary error indicators
        """
        def neg_log_likelihood(params):
            alpha, beta = params

            # Compute predicted probabilities
            logits = alpha * scores + beta
            probs = expit(logits)

            # Binary cross-entropy
            eps = 1e-10
            probs = np.clip(probs, eps, 1 - eps)
            nll = -np.mean(
                errors * np.log(probs) + (1 - errors) * np.log(1 - probs)
            )

            # L2 regularization
            if self.regularization > 0:
                nll += self.regularization * (alpha ** 2 + beta ** 2)

            return nll

        # Initial values
        alpha_init = self.alpha if self.learn_alpha else 1.0
        beta_init = self.beta if self.learn_beta else 0.0

        # Optimize
        if self.learn_alpha and self.learn_beta:
            # Learn both
            result = minimize(
                neg_log_likelihood,
                x0=[alpha_init, beta_init],
                method='L-BFGS-B'
            )
            self.alpha, self.beta = result.x

        elif self.learn_alpha:
            # PHC-TS: only alpha
            def nll_alpha(alpha):
                return neg_log_likelihood([alpha[0], 0.0])
            result = minimize(nll_alpha, x0=[alpha_init], method='L-BFGS-B')
            self.alpha = result.x[0]
            self.beta = 0.0

        elif self.learn_beta:
            # PHC-BO: only beta
            def nll_beta(beta):
                return neg_log_likelihood([1.0, beta[0]])
            result = minimize(nll_beta, x0=[beta_init], method='L-BFGS-B')
            self.alpha = 1.0
            self.beta = result.x[0]

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to get calibrated error probabilities.

        Args:
            scores: (N,) array of uncertainty scores

        Returns:
            (N,) array of calibrated error probabilities
        """
        logits = self.alpha * scores + self.beta
        return expit(logits)

    def get_params(self) -> dict:
        """Return learned parameters."""
        return {"alpha": self.alpha, "beta": self.beta}


class TemperatureScaling(PlattScaling):
    """
    Temperature Scaling (PHC-TS) calibration.

    Special case of Platt scaling with beta=0:
        P(error | score) = sigmoid(alpha * score)

    Only learns the temperature parameter alpha.
    """

    def __init__(self, regularization: float = 0.0):
        super().__init__(learn_alpha=True, learn_beta=False, regularization=regularization)


class BiasOnlyCalibration(PlattScaling):
    """
    Bias Only (PHC-BO) calibration.

    Special case of Platt scaling with alpha=1:
        P(error | score) = sigmoid(score + beta)

    Only learns the bias parameter beta.
    """

    def __init__(self, regularization: float = 0.0):
        super().__init__(learn_alpha=False, learn_beta=True, regularization=regularization)


class UniformMassCalibration(ScoreCalibrator):
    """
    Uniform Mass Calibration (Quantile Binning).

    Divides data into bins with equal number of samples (quantile binning),
    then computes empirical error rate in each bin.

    For a new score, finds the appropriate bin and returns the
    empirical error rate of that bin.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize uniform mass calibrator.

        Args:
            n_bins: Number of bins (default: 10)
        """
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_error_rates = None

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> None:
        """
        Compute bin edges (quantiles) and empirical error rates.

        Args:
            scores: (N,) array of uncertainty scores
            errors: (N,) array of binary error indicators
        """
        # Compute quantile-based bin edges
        quantiles = np.linspace(0, 100, self.n_bins + 1)
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
