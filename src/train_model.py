import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from .custom_metrics import auc_pr, auc_pr_alt
from .feature_selection import make_feature_selection
from .model import BidWinModel
from .model_optimisation import optimize_model


def train_model(
    train_data: pd.DataFrame,
    target_column: str = "target",
    with_hp_opt: bool = True,
    with_feature_selection: bool = True,
    n_trials: int = 100,
    model_config_path: str = "bid_win_model_params.yaml",
    model_save_path: str = "bid_win_model.joblib",
) -> None:
    """
    Trains a machine learning model for predicting bid win probabilities.

    Args:
        train_data: The dataset used for training, containing features and the target variable.
        target_column: The name of the target column in `train_data` indicating whether a bid won or lost.
        with_hp_opt: If True, hyperparameter optimization will be performed during model training.
        with_feature_selection: If True, feature selection will be applied to improve model performance.
        n_trials: The number of trials to perform during hyperparameter optimization. Ignored if `with_hp_opt` is False.
        model_config_path: The path to a YAML file containing model configuration parameters.
        model_save_path: The file path to save the trained model.
    """

    train_data = BidWinModel.prepare_and_extract_features(train_data)

    features = train_data.drop(columns=[target_column]).columns.tolist()
    categorical_features = (
        train_data.drop(columns=[target_column])
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )

    # If no config file is available, we will need to run hyperparameter optimization
    model_config_path = "bid_win_model_params.yaml"
    if not os.path.exists(model_config_path):
        with_hp_opt = True

    if with_hp_opt or with_feature_selection:
        # to speed up calculations, let's use the sample of the data
        _, reduced_train_data = train_test_split(
            train_data,
            test_size=min(train_data.shape[0], 1000000),
            random_state=42,
            stratify=train_data[target_column],
        )

    if with_hp_opt:
        tunned_model, tunned_params = optimize_model(
            reduced_train_data,
            features,
            categorical_features,
            "bid_win_model_optimisation",
            "auc_pr_alt",
            target_column_name=target_column,
            n_trials=n_trials,
            plot_report=True,
        )
    else:
        with open(model_config_path, "r") as f:
            model_params_dict = yaml.safe_load(f)
        tunned_params = model_params_dict["tunned_params"]
        features = model_params_dict["features"]
        categorical_features = model_params_dict["categorical_features"]

        tunned_model = BidWinModel(tunned_params)

    if with_feature_selection:
        features_to_drop = make_feature_selection(
            tunned_model,
            reduced_train_data[features],
            reduced_train_data[target_column],
            categorical_features,
            metric=auc_pr_alt,
            greater_is_better=True,
        )

        features = [fe for fe in features if fe not in features_to_drop]
        categorical_features = [
            fe for fe in categorical_features if fe not in features_to_drop
        ]

    model_params_dict = {}
    model_params_dict["tunned_params"] = tunned_params
    model_params_dict["features"] = features
    model_params_dict["categorical_features"] = categorical_features

    with open(model_config_path, "w") as f:
        yaml.dump(model_params_dict, f)

    # Fit tunned model with all available data
    final_model = tunned_model.fit(train_data[features], train_data[target_column])

    joblib.dump(final_model, model_save_path)
