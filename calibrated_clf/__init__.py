"""
Binary Classification Framework with Advanced Calibration

Public API — import directly from `calibrated_clf`:

    from calibrated_clf import CalibratedBinaryClassifier, load_fraud_data, SimpleSplitter
"""

from .model import BidWinModel, CalibratedBinaryClassifier
from .calibration import MultiCalibrationWrapper, VennABERSBinaryCalibrator
from .data_loader import create_time_groups, get_categorical_features, load_fraud_data
from .data_transformers import TimeWindowedTargetEncoder, FraudFeatureEngineer
from .validators import TemporalGroupSplitter, SimpleSplitter
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
    "FraudFeatureEngineer",
    "TemporalGroupSplitter",
    "SimpleSplitter",
    "optimize_model",
    "train_model",
]
