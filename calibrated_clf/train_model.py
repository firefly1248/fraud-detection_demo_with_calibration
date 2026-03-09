import os
import typing as tp

import joblib
import pandas as pd
import yaml

from .config import RANDOM_SEED
from .custom_metrics import auc_pr_alt
from .feature_selection import make_feature_selection
from .model import CalibratedBinaryClassifier
from .model_optimisation import optimize_model


def train_model(
    train_data: pd.DataFrame,
    target_column: str = "isFraud",
    with_hp_opt: bool = True,
    with_feature_selection: bool = True,
    n_trials: int = 100,
    calibration_method: str = "isotonic",
    calibration_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    model_config_path: str = "model_params.yaml",
    model_save_path: str = "model.joblib",
) -> CalibratedBinaryClassifier:
    """
    Train a CalibratedBinaryClassifier with optional HP optimisation and feature selection.

    Args:
        train_data: Dataset with features and target column.
        target_column: Name of the binary target column.
        with_hp_opt: Run Optuna hyperparameter optimisation if True.
        with_feature_selection: Apply recursive feature elimination if True.
        n_trials: Number of Optuna trials (ignored when with_hp_opt=False).
        calibration_method: Calibration method for the final model
            ('isotonic', 'venn_abers', 'sigmoid', 'none').
        calibration_params: Extra kwargs forwarded to MultiCalibrationWrapper
            (e.g. {'cal_size': 0.2, 'time_column': 'TransactionDT'}).
        model_config_path: Path to YAML file for saving/loading model params.
        model_save_path: Path to save the final fitted model (.joblib).

    Returns:
        Fitted CalibratedBinaryClassifier.
    """
    calibration_params = calibration_params or {}

    train_data = CalibratedBinaryClassifier.prepare_and_extract_features(train_data)

    features = train_data.drop(columns=[target_column]).columns.tolist()
    categorical_features = (
        train_data.drop(columns=[target_column])
        .select_dtypes(include=["object", "category", "string"])
        .columns.tolist()
    )

    # If no config file is available, force hyperparameter optimisation
    if not os.path.exists(model_config_path):
        with_hp_opt = True

    # Use a capped sample for HP optimisation and feature selection (speed)
    if with_hp_opt or with_feature_selection:
        from sklearn.model_selection import train_test_split

        sample_size = min(train_data.shape[0], 1_000_000)
        _, reduced_data = train_test_split(
            train_data,
            test_size=sample_size,
            random_state=RANDOM_SEED,
            stratify=train_data[target_column],
        )

    if with_hp_opt:
        study_name = os.path.splitext(os.path.basename(model_config_path))[0] + "_optimisation"
        _, tunned_params = optimize_model(
            reduced_data,
            features,
            categorical_features,
            study_name=study_name,
            metric_name="auc_pr_alt",
            target_column_name=target_column,
            n_trials=n_trials,
            plot_report=True,
        )
    else:
        with open(model_config_path, "r") as f:
            saved = yaml.safe_load(f)
        tunned_params = saved["tunned_params"]
        features = saved["features"]
        categorical_features = saved["categorical_features"]

    if with_feature_selection:
        # Fit a temporary model on the reduced dataset to rank features
        tmp_model = CalibratedBinaryClassifier(
            variable_params=tunned_params, calibration_method="none"
        )
        features_to_drop = make_feature_selection(
            tmp_model,
            reduced_data[features],
            reduced_data[target_column],
            categorical_features,
            metric=auc_pr_alt,
            greater_is_better=True,
        )
        features = [f for f in features if f not in features_to_drop]
        categorical_features = [f for f in categorical_features if f not in features_to_drop]

    # Persist config so the next run can skip optimisation
    with open(model_config_path, "w") as f:
        yaml.dump(
            {
                "tunned_params": tunned_params,
                "features": features,
                "categorical_features": categorical_features,
                "calibration_method": calibration_method,
            },
            f,
        )

    # Fit the final model on all available data with the chosen calibration
    final_model = CalibratedBinaryClassifier(
        variable_params=tunned_params,
        calibration_method=calibration_method,
        calibration_params=calibration_params,
    )
    final_model.fit(train_data[features], train_data[target_column])

    joblib.dump(final_model, model_save_path)
    return final_model
