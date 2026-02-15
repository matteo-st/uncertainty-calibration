"""
Uncertainty Calibration for LLM Text Classification

Implementation of calibration techniques from:
"Unsupervised Calibration through Prior Adaptation for Text Classification
using Large Language Models" (Estienne et al., 2023)
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
from .evaluation import compute_metrics

__all__ = [
    "LLMClassifier",
    "load_agnews",
    "ClassificationDataset",
    "Calibrator",
    "ContentFreeCalibrator",
    "UCPANaiveCalibrator",
    "UCPACalibrator",
    "SUCPACalibrator",
    "compute_metrics",
]
