import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

plt.rcParams["figure.figsize"] = (10, 6)


def select_threshold_maximize_f1_cv(model, X, y, n_splits=5):
    """
    Selects the threshold to maximize the F1 score based on 5-fold cross-validation.
    Also plots the F1 score vs. threshold curve for each fold and the mean F1 score across all folds.

    Parameters:
    - model: a scikit-learn estimator - The model to be trained.
    - X: array-like of shape (n_samples, n_features) - The input samples.
    - y: array-like of shape (n_samples,) - The true labels.
    - n_splits: int, default=5 - Number of folds for cross-validation.

    Returns:
    - best_threshold: float - The optimal threshold to maximize the F1 score.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    thresholds = np.arange(0.01, 1.01, 0.01)

    best_threshold = 0.0
    best_f1 = 0.0
    all_f1_scores = np.zeros((n_splits, len(thresholds)))

    for fold_index, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        for i, threshold in enumerate(thresholds):
            y_val_pred = (y_val_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            all_f1_scores[fold_index, i] = f1

    mean_f1_scores = np.mean(all_f1_scores, axis=0)
    best_threshold = thresholds[np.argmax(mean_f1_scores)]
    best_f1 = np.max(mean_f1_scores)

    for fold_index in range(n_splits):
        plt.plot(thresholds, all_f1_scores[fold_index], label=f"Fold {fold_index + 1}")

    # Plotting the mean F1 score vs. threshold
    plt.plot(thresholds, mean_f1_scores, label="Mean F1 Score", color="black", linewidth=2)
    plt.axvline(
        x=best_threshold,
        color="r",
        linestyle="--",
        label=f"Best Threshold: {best_threshold:.2f} (F1={best_f1:.4f})",
    )
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Threshold (5-Fold Cross-Validation)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_threshold


def plot_f1_score_thresholds(y_true, y_pred_proba, provided_threshold):
    """
    Plots the F1 score for thresholds from 0.01 to 1 with a step of 0.01 and highlights the provided threshold.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels.
    - y_pred_proba: array-like of shape (n_samples,) - Predicted probabilities.
    - provided_threshold: float - The specified threshold for which the F1 score should be highlighted.
    """
    thresholds = np.arange(0.01, 1.01, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    provided_f1 = f1_score(y_true, (y_pred_proba >= provided_threshold).astype(int))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1 Score")
    plt.axvline(
        x=provided_threshold,
        color="blue",
        linestyle="--",
        label=f"F1 Score: {provided_f1:.4f} at threshold {provided_threshold:.2f}",
    )
    plt.scatter([provided_threshold], [provided_f1], color="blue")
    plt.title("F1 Score vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_calibration_curve(y_true, y_pred_proba, n_bins=20):
    """
    Plots the calibration curve.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels.
    - y_pred_proba: array-like of shape (n_samples,) - Predicted probabilities.
    - n_bins: int, default=10 - Number of bins to discretize the [0, 1] interval.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_feature_importances(model):
    imp_df = pd.DataFrame({"feature": model.features, "importance": model.feature_importances_})

    imp_df.sort_values("importance", inplace=True)

    sns.barplot(x="feature", y="importance", data=imp_df)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
