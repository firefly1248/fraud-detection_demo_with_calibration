import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def auc_pr(y_true, y_scores):
    # precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # auc_score = auc(recall, precision)
    auc_score = average_precision_score(y_true, y_scores)
    return auc_score


def auc_pr_alt(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_score = auc(recall, precision)
    return auc_score


def get_precision_recall_curves(y_true, y_scores):
    precisions = []
    recalls = []
    all_thresholds = np.arange(0, 1.01, 0.01).tolist()
    for th in all_thresholds:
        pred_at_th = (y_scores >= th).astype(int)
        precisions.append(precision_score(y_true, pred_at_th, zero_division=0))
        recalls.append(recall_score(y_true, pred_at_th, zero_division=0))

    return pd.DataFrame({"precision": precisions, "recall": recalls}, index=all_thresholds)
