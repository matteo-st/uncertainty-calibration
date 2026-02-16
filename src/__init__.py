"""
Uncertainty Calibration for LLM Text Classification

Implementation of calibration techniques from:
1. "Unsupervised Calibration through Prior Adaptation for Text Classification
   using Large Language Models" (Estienne et al., 2023) - LLM posterior calibration
2. "Adapting Language Models to Produce Good Class Probabilities for
   Classification Tasks" (Estienne et al., 2026, TMLR) - Post-hoc calibration

This package also implements uncertainty score calibration for error prediction.
"""

from .models import LLMClassifier
from .data import load_agnews, ClassificationDataset
from .calibration import (
    Calibrator,
    ContentFreeCalibrator,
    UCPANaiveCalibrator,
    UCPACalibrator,
    SUCPACalibrator,
)
from .evaluation import (
    compute_metrics,
    compute_rocauc,
    compute_ece,
    compute_binary_cross_entropy,
    compute_error_prediction_metrics,
)
from .uncertainty import (
    compute_uncertainty_scores,
    compute_max_proba_complement,
    compute_margin,
    compute_doctor,
    compute_doctor_normalized,
    prepare_calibration_data,
)
from .score_calibration import (
    ScoreCalibrator,
    NoCalibration,
    PlattScaling,
    TemperatureScaling,
    BiasOnlyCalibration,
    UniformMassCalibration,
    IsotonicCalibration,
    get_calibrator,
)

__all__ = [
    # Models
    "LLMClassifier",
    # Data
    "load_agnews",
    "ClassificationDataset",
    # LLM Calibration
    "Calibrator",
    "ContentFreeCalibrator",
    "UCPANaiveCalibrator",
    "UCPACalibrator",
    "SUCPACalibrator",
    # Classification Evaluation
    "compute_metrics",
    # Uncertainty Scores
    "compute_uncertainty_scores",
    "compute_max_proba_complement",
    "compute_margin",
    "compute_doctor",
    "compute_doctor_normalized",
    "prepare_calibration_data",
    # Score Calibration
    "ScoreCalibrator",
    "NoCalibration",
    "PlattScaling",
    "TemperatureScaling",
    "BiasOnlyCalibration",
    "UniformMassCalibration",
    "IsotonicCalibration",
    "get_calibrator",
    # Error Prediction Evaluation
    "compute_rocauc",
    "compute_ece",
    "compute_binary_cross_entropy",
    "compute_error_prediction_metrics",
]
