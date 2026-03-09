"""
Binary Classification Framework with Advanced Calibration

Public API — import directly from `src`:

    from src import CalibratedBinaryClassifier, load_fraud_data, SimpleSplitter
"""

from .model import BidWinModel, CalibratedBinaryClassifier
from .calibration import MultiCalibrationWrapper, VennABERSBinaryCalibrator
from .data_loader import create_time_groups, get_categorical_features, load_fraud_data
from .data_transformers import TimeWindowedTargetEncoder
from .model_optimisation import optimize_model
from .train_model import train_model

__all__ = [
    "CalibratedBinaryClassifier",
    "BidWinModel",
    "MultiCalibrationWrapper",
    "VennABERSBinaryCalibrator",
    "load_fraud_data",
    "create_time_groups",
    "get_categorical_features",
    "TimeWindowedTargetEncoder",
    "optimize_model",
    "train_model",
]
