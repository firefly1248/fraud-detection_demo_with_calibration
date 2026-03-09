import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_SEED, VALIDATION_N_SPLITS
from .custom_metrics import auc_pr, auc_pr_alt
from .model import CalibratedBinaryClassifier


def optimize_model(
    train_val_data: pd.DataFrame,
    features: list,
    categorical_features: list,
    study_name: str,
    metric_name: str,
    target_column_name: str = "target",
    n_trials: int = 100,
    plot_report: bool = True,
) -> tuple[CalibratedBinaryClassifier, dict]:
    """
    Optimizes a machine learning model using hyperparameter tuning with Optuna for a given dataset.

    Args:
        train_val_data: The dataset used for training and validation, containing features and target labels.
        features: A list of feature column names used for training the model.
        categorical_features: A list of categorical feature column names to be handled appropriately during model training.
        study_name: The name of the Optuna study for hyperparameter optimization.
        metric_name: The name of the evaluation metric to optimize.
        target_column_name: The name of the target column in the dataset.
        n_trials: The number of trials for the Optuna hyperparameter optimization.
        plot_report: If True, generates and displays a report for the optimization process.

    Returns:
        tuple[CalibratedBinaryClassifier, dict]:
            - CalibratedBinaryClassifier: The best model obtained after optimization.
            - dict: A dictionary containing the best hyperparameters.
    """

    def objective(trial):

        features_trial = features.copy()

        train_params = {
            "boosting_type": trial.suggest_categorical(
                "classifier__boosting_type", ["gbdt", "goss", "dart"]
            ),
            "learning_rate": trial.suggest_float(
                "classifier__learning_rate", 1e-3, 1.0, log=True
            ),
            "max_depth": trial.suggest_int("classifier__max_depth", 3, 8),
            # "objective": trial.suggest_categorical("classifier__objective", ["binary", "cross_entropy"]),
            "reg_alpha": trial.suggest_float(
                "classifier__reg_alpha", 1e-8, 1e4, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "classifier__reg_lambda", 1e-8, 1e4, log=True
            ),
            "n_estimators": trial.suggest_int("classifier__n_estimators", 50, 200),
            # "linear_tree ": trial.suggest_categorical("classifier__linear_tree", [True, False]),
            # "max_bins": trial.suggest_int("classifier__max_bins", 32, 512, log=True),
            "min_split_gain": trial.suggest_float(
                "classifier__min_split_gain", 1e-8, 1.0, log=True
            ),
            "colsample_bytree": trial.suggest_float(
                "classifier__colsample_bytree", 0.3, 1.0
            ),
            "num_leaves": trial.suggest_int("classifier__num_leaves", 16, 128),
        }

        if train_params["boosting_type"] == "goss":
            train_params["top_rate"] = trial.suggest_float("classifier__top_rate", 0.1, 0.5)
            train_params["other_rate"] = trial.suggest_float("classifier__other_rate", 0.01, 0.5)

        elif train_params["boosting_type"] == "dart":
            train_params["drop_rate"] = trial.suggest_float("classifier__drop_rate", 0.01, 0.25)

        if len(categorical_features) > 0:
            cat_features_strategy = trial.suggest_categorical(
                "cat_encoder__strategy",
                [
                    "catboost",
                    #'glmme',
                    "james_stein",
                    #'backward_difference',
                    'm_estimate',
                    #'polynomial',
                    "target_encoder",
                    # "ordinal",
                ],
            )

        trial_metrics = []

        splitter = StratifiedKFold(
            n_splits=VALIDATION_N_SPLITS, random_state=RANDOM_SEED, shuffle=True
        )
        for train_index, test_index in splitter.split(
            train_val_data, train_val_data[target_column_name]
        ):
            train_set = train_val_data.iloc[train_index].reset_index(drop=True)

            val_set = train_val_data.iloc[test_index].reset_index(drop=True)

            X_train = train_set[features_trial]
            X_val = val_set[features_trial]
            y_train = train_set[target_column_name]
            y_val = val_set[target_column_name]

            model = CalibratedBinaryClassifier(trial.params)

            model.fit(X_train, y_train)

            y_pred_val = model.predict_proba(X_val)[:, 1]
            # pd.Series(y_pred_val).hist(bins=25)
            # plt.show()

            if metric_name == "brier":
                trial_metrics.append(brier_score_loss(y_val, y_pred_val))
            elif metric_name == "auc_pr":
                trial_metrics.append(auc_pr(y_val, y_pred_val))
            elif metric_name == "auc_pr_alt":
                trial_metrics.append(auc_pr_alt(y_val, y_pred_val))
            elif metric_name == "logloss":
                trial_metrics.append(log_loss(y_val, y_pred_val))
            elif metric_name == "roc_auc":
                trial_metrics.append(roc_auc_score(y_val, y_pred_val))
            else:
                raise Exception("Unknown metric")

        resulting_trial_metric = np.average(trial_metrics)  # - np.std(trial_metrics)
        return resulting_trial_metric

    direction = "minimize" if metric_name in ["brier", "logloss"] else "maximize"
    storage_name = "sqlite:///{}.db".format(study_name)
    try:
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(),
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )
        print(f"📁 Study stored in {study_name}.db")
    except Exception as e:
        print(f"⚠️  SQLite storage unavailable ({e}), using in-memory storage")
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(),
            study_name=study_name,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    if plot_report:
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.yscale("log")
        plt.show()

        for param in study.best_params.keys():
            optuna.visualization.matplotlib.plot_slice(study, params=[param])
            plt.yscale("log")
            plt.show()

    best_params = study.best_trial.params.copy()

    model = CalibratedBinaryClassifier(best_params)

    model.fit(train_val_data[features], train_val_data[target_column_name])

    return model, best_params
