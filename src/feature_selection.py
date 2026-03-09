import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


def make_feature_selection(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list,
    metric: callable = average_precision_score,
    n_splits: int = 5,
    greater_is_better: bool = True,
    fs_tolerance=-1e-5,
) -> list:

    features = X.columns.tolist()

    features_to_drop = []
    features_to_keep = []
    sorted_features_list = None
    already_checked_features = []
    candidate_to_drop = None
    best_metric = None
    n_features = len(features)

    kf = StratifiedKFold(n_splits=n_splits)
    while set(features) != set(already_checked_features):
        cv_metrics = []

        if isinstance(sorted_features_list, type(None)):
            features_round = [fe for fe in features if fe not in set(features_to_drop)]
            categorical_features_round = [fe for fe in categorical_features if fe in features_round]

        else:
            for fe in sorted_features_list:
                if fe not in set(already_checked_features):
                    candidate_to_drop = fe
                    break
            features_round = [
                fe for fe in features if fe not in features_to_drop + [candidate_to_drop]
            ]
            categorical_features_round = [fe for fe in categorical_features if fe in features_round]

        feature_scores = np.zeros(len(features_round))

        for train_index, test_index in kf.split(X, y):

            X_train, X_test = (
                X.iloc[train_index][features_round],
                X.iloc[test_index][features_round],
            )
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            candidate_model = clone(model)

            candidate_model.fit(X_train, y_train)

            y_test_pred = candidate_model.predict_proba(X_test)[:, 1]

            cv_metrics.append(metric(y_test, y_test_pred))

            feature_scores += candidate_model.feature_importances_

        metric_round = np.average(cv_metrics)

        if isinstance(sorted_features_list, type(None)):
            best_metric = metric_round
            fe_imp_df = pd.DataFrame({"importance": (feature_scores / n_splits).tolist()})
            fe_imp_df.index = features_round
            sorted_features_list = fe_imp_df.sort_values(
                by=["importance"], ascending=True
            ).index.tolist()
            print("Best metric before selection: {:.4f}".format(metric_round))

        else:
            diff_metrics = metric_round - best_metric
            print(
                "{}/{}. Gain in metric at the drop of {} = {:.4f}, metric = {:.4f}".format(
                    len(already_checked_features) + 1,
                    n_features,
                    candidate_to_drop,
                    diff_metrics,
                    metric_round,
                )
            )

            if (greater_is_better and diff_metrics < fs_tolerance) or (
                not greater_is_better and diff_metrics > fs_tolerance
            ):
                print(f"keep: {candidate_to_drop}")
                features_to_keep.append(candidate_to_drop)
            else:
                print(f"drop: {candidate_to_drop}")
                best_metric = metric_round
                features_to_drop.append(candidate_to_drop)
                fe_imp_df = pd.DataFrame({"importance": (feature_scores / n_splits).tolist()})
                fe_imp_df.index = features_round
                sorted_features_list = fe_imp_df.sort_values(
                    by=["importance"], ascending=True
                ).index.tolist()

            already_checked_features.append(candidate_to_drop)

    return features_to_drop
