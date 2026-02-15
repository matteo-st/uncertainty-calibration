"""
Utility functions for the calibration experiments.
"""

import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def save_probabilities(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    metadata: Optional[Dict] = None,
):
    """
    Save probability predictions to numpy file.

    Args:
        probs: Probability array
        labels: Label array
        output_path: Path for .npz file
        metadata: Optional metadata dict
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {"probs": probs, "labels": labels}
    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                save_dict[k] = np.array([v])

    np.savez(output_path, **save_dict)


def load_probabilities(probs_path: str) -> Dict[str, np.ndarray]:
    """Load probability predictions from numpy file."""
    data = np.load(probs_path, allow_pickle=True)
    return dict(data)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
