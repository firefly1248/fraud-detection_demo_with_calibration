"""
Binary Classification Model with Advanced Calibration

This module provides a scikit-learn compatible binary classifier with:
- Automated feature engineering for fraud detection and bid-win prediction
- Multiple calibration methods (isotonic, Venn-ABERS, sigmoid)
- SHAP value computation for interpretability
- Support for categorical encoding and missing value handling

Example:
    >>> from src.model import CalibratedBinaryClassifier
    >>> from src.data_loader import load_fraud_data
    >>>
    >>> # Load data
    >>> df = load_fraud_data(sample_frac=0.1)
    >>> X = df.drop(columns=['isFraud'])
    >>> y = df['isFraud']
    >>>
    >>> # Configure model
    >>> params = {
    >>>     'classifier__learning_rate': 0.05,
    >>>     'classifier__max_depth': 6,
    >>>     'cat_encoder__strategy': 'target_encoder'
    >>> }
    >>>
    >>> # Train with Venn-ABERS calibration
    >>> model = CalibratedBinaryClassifier(
    >>>     variable_params=params,
    >>>     calibration_method='venn_abers'
    >>> )
    >>> model.fit(X, y)
    >>>
    >>> # Predict with uncertainty intervals
    >>> predictions = model.predict_proba_with_intervals(X_test)
"""

import typing as tp
from functools import wraps

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from .config import MODEL_FIXED_PARAMS
from .data_transformers import CatFeaturesEncoder, FieldsToCategory
from .calibration import MultiCalibrationWrapper

# Type aliases for clarity
ModelParams = tp.Dict[str, tp.Any]
Features = tp.List[str]
PredictionIntervals = tp.Dict[str, np.ndarray]


def check_is_fitted(func: tp.Callable) -> tp.Callable:
    """
    Decorator to check if model is fitted before prediction.

    Args:
        func: Method to decorate (predict, predict_proba, etc.)

    Returns:
        Wrapped function that checks fit status

    Raises:
        ValueError: If model is not fitted
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_fitted_", False):
            raise ValueError(f"Model is not fitted yet. Call fit() before using {func.__name__}().")
        return func(self, *args, **kwargs)

    return wrapper


class CalibratedBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible binary classifier with advanced calibration.

    This classifier wraps LightGBM with:
    - Automated feature engineering (fraud detection & bid-win)
    - Categorical encoding (target encoding, CatBoost, etc.)
    - Multiple calibration methods (isotonic, Venn-ABERS, sigmoid)
    - SHAP value computation
    - Uncertainty quantification (Venn-ABERS intervals)

    Parameters
    ----------
    variable_params : dict
        Hyperparameters for the model pipeline. Keys should be prefixed:
        - 'classifier__*': LightGBM parameters (learning_rate, max_depth, etc.)
        - 'cat_encoder__*': Categorical encoder parameters (strategy, etc.)

    calibration_method : str, default='isotonic'
        Calibration method to use:
        - 'isotonic': Isotonic regression (sklearn standard)
        - 'venn_abers': Venn-ABERS conformal prediction (provides intervals)
        - 'sigmoid': Platt scaling (logistic regression)
        - 'none' or None: No calibration

    calibration_params : dict, optional
        Additional parameters for calibration method:
        - For Venn-ABERS: {'cal_size': 0.2, 'random_state': 0}
        - For cross-validation: {'cv_folds': 5}

    Attributes
    ----------
    model_ : sklearn.pipeline.Pipeline
        Fitted pipeline with categorical encoding and classifier

    calibrated_model_ : MultiCalibrationWrapper
        Calibrated version of the model

    features_ : list of str
        Feature names used during training

    feature_importances_ : np.ndarray
        Feature importance scores from LightGBM

    classes_ : np.ndarray
        Class labels

    is_fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    Basic usage with isotonic calibration:

    >>> model = CalibratedBinaryClassifier(
    >>>     variable_params={'classifier__learning_rate': 0.05},
    >>>     calibration_method='isotonic'
    >>> )
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict_proba(X_test)

    Advanced usage with Venn-ABERS:

    >>> model = CalibratedBinaryClassifier(
    >>>     variable_params={'classifier__learning_rate': 0.05},
    >>>     calibration_method='venn_abers',
    >>>     calibration_params={'cal_size': 0.2}
    >>> )
    >>> model.fit(X_train, y_train)
    >>> intervals = model.predict_proba_with_intervals(X_test)
    >>> print(f"Mean uncertainty: {intervals['interval_width'].mean():.4f}")

    Notes
    -----
    - Feature engineering is applied automatically during fit() and predict()
    - Categorical features are detected automatically
    - Missing values are handled by LightGBM
    - For fraud detection, use dataset with 'TransactionAmt', 'card1', etc.
    - For bid-win, use dataset with 'price', 'flr', 'dsp', etc.

    See Also
    --------
    MultiCalibrationWrapper : Calibration wrapper supporting multiple methods
    prepare_and_extract_features : Feature engineering function
    """

    def __init__(
        self,
        variable_params: ModelParams,
        calibration_method: str = "isotonic",
        calibration_params: tp.Optional[ModelParams] = None,
    ) -> None:
        self.variable_params = variable_params
        self.calibration_method = calibration_method
        self.calibration_params = calibration_params or {}

        # Attributes set during fit (trailing underscore per sklearn convention)
        self.model_: tp.Optional[Pipeline] = None
        self.calibrated_model_: tp.Optional[MultiCalibrationWrapper] = None
        self.features_: tp.Optional[Features] = None
        self.feature_importances_: tp.Optional[np.ndarray] = None
        self.classes_: tp.Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    @staticmethod
    def build_model(
        variable_params: ModelParams,
        categorical_features: Features,
    ) -> Pipeline:
        """
        Build model pipeline with categorical encoding and classifier.

        Parameters
        ----------
        variable_params : dict
            Hyperparameters with module prefixes:
            - 'classifier__*': LightGBM parameters
            - 'cat_encoder__*': Categorical encoder parameters

        categorical_features : list of str
            Names of categorical features for encoding

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline with steps: [cat_encoder, ordinal_to_category, classifier]

        Notes
        -----
        - If no categorical features, cat_encoder step is skipped
        - If using ordinal encoding, adds conversion to category dtype
        - Classifier is always LightGBMClassifier with fixed params from config
        """
        cat_encoder_params = {}
        classifier_params = {}

        # Separate parameters by module
        for key, value in variable_params.items():
            if "__" in key:
                module_name, param_name = key.split("__", 1)
                if module_name == "classifier":
                    classifier_params[param_name] = value
                elif module_name == "cat_encoder":
                    cat_encoder_params[param_name] = value

        # Merge with fixed parameters
        classifier_params.update(MODEL_FIXED_PARAMS)
        cat_encoder_params.update({"cols": categorical_features})

        # Build pipeline steps
        steps = []

        if len(categorical_features) > 0:
            steps.append(("cat_encoder", CatFeaturesEncoder(**cat_encoder_params)))

            # Convert to category dtype if using ordinal encoding
            if cat_encoder_params.get("strategy") == "ordinal":
                steps.append(
                    ("ordinal_to_category", FieldsToCategory(variables=categorical_features))
                )

        steps.append(("classifier", LGBMClassifier(**classifier_params)))

        return Pipeline(steps=steps)

    @staticmethod
    def prepare_and_extract_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply automated feature engineering for fraud detection and bid-win.

        This method detects the dataset type based on available columns and
        applies appropriate feature transformations. Supports both fraud
        detection (IEEE-CIS) and bid-win prediction datasets.

        Parameters
        ----------
        X : pd.DataFrame
            Input features (without target column)

        Returns
        -------
        pd.DataFrame
            Transformed features with engineered columns added

        Features Created
        ----------------
        **Fraud Detection** (if 'TransactionAmt' present):
        - TransactionAmt_log: Log-transformed amount
        - TransactionAmt_decimal: Decimal part of amount
        - TransactionAmt_mean_card1: Mean amount per card
        - TransactionAmt_std_card1: Std deviation per card
        - TransactionDT_hour: Hour of day (0-23)
        - TransactionDT_day: Day number since start
        - TransactionDT_weekday: Day of week (0-6)
        - email_domain_match: Whether P/R domains match
        - email_domain_missing: Whether email domains are missing
        - dist_ratio: Ratio of dist1 to dist2
        - D_missing_count: Count of missing D-features
        - addr_match: Whether addr1 == addr2
        - M_true_count: Count of M-features == 'T'

        **Bid-Win Prediction** (if 'price' present):
        - lang: Cleaned language code
        - price_to_flr_ratio: Price to floor ratio
        - price_to_sellerClearPrice_ratio: Price to clear price ratio
        - avg_price_by_hour: Average price per hour
        - price_std_by_dsp: Price std deviation by DSP
        - avg_price_by_dsp: Average price by DSP
        - screen_ratio: Screen width/height ratio
        - screen_diagonal: Screen diagonal size

        Examples
        --------
        >>> X_engineered = CalibratedBinaryClassifier.prepare_and_extract_features(X)
        >>> print(f"Original: {X.shape[1]} features")
        >>> print(f"Engineered: {X_engineered.shape[1]} features")

        Notes
        -----
        - This is a static method and can be called without fitting
        - Feature engineering is applied automatically in fit() and predict()
        - Missing values are preserved (handled by LightGBM)
        - Groupby aggregations may introduce NaNs for single-element groups
        """
        X_ = X.copy()

        # === FRAUD DETECTION FEATURES ===
        if "TransactionAmt" in X_.columns:
            # Transaction amount transformations
            X_["TransactionAmt_log"] = np.log1p(X_["TransactionAmt"])
            X_["TransactionAmt_decimal"] = X_["TransactionAmt"] - X_["TransactionAmt"].astype(int)

        if "card1" in X_.columns and "TransactionAmt" in X_.columns:
            # Card-level aggregations (fraud patterns)
            X_["TransactionAmt_mean_card1"] = X_.groupby("card1")["TransactionAmt"].transform(
                "mean"
            )
            X_["TransactionAmt_std_card1"] = X_.groupby("card1")["TransactionAmt"].transform("std")

        if "TransactionDT" in X_.columns:
            # Time-based features
            X_["TransactionDT_hour"] = (X_["TransactionDT"] % 86400) // 3600
            X_["TransactionDT_day"] = X_["TransactionDT"] // 86400
            X_["TransactionDT_weekday"] = (X_["TransactionDT"] // 86400) % 7

        if "P_emaildomain" in X_.columns and "R_emaildomain" in X_.columns:
            # Email domain features
            X_["email_domain_match"] = (X_["P_emaildomain"] == X_["R_emaildomain"]).astype(int)
            X_["email_domain_missing"] = (
                X_["P_emaildomain"].isnull() | X_["R_emaildomain"].isnull()
            ).astype(int)

        if "dist1" in X_.columns and "dist2" in X_.columns:
            # Distance ratio with safety for division by zero
            X_["dist_ratio"] = X_["dist1"] / (X_["dist2"] + 1e-5)

        # D-feature missingness (timedelta features)
        d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in X_.columns]
        if len(d_cols) > 0:
            X_["D_missing_count"] = X_[d_cols].isnull().sum(axis=1)

        if "addr1" in X_.columns and "addr2" in X_.columns:
            # Address matching
            X_["addr_match"] = (X_["addr1"] == X_["addr2"]).astype(int)

        # M-feature aggregation (boolean match features)
        m_cols = [f"M{i}" for i in range(1, 10) if f"M{i}" in X_.columns]
        if len(m_cols) > 0:
            X_["M_true_count"] = (X_[m_cols] == "T").sum(axis=1)

        # === BID-WIN FEATURES (legacy support) ===
        if "lang" in X_.columns:
            # Language code cleaning
            X_["lang"] = X_["lang"].apply(
                lambda x: x.split("_")[0].split("-")[0] if isinstance(x, str) else x
            )

        if "price" in X_.columns and "flr" in X_.columns:
            X_["price_to_flr_ratio"] = X_["price"] / X_["flr"]

        if "price" in X_.columns and "sellerClearPrice" in X_.columns:
            X_["price_to_sellerClearPrice_ratio"] = X_["price"] / X_["sellerClearPrice"]

        if "hour" in X_.columns and "price" in X_.columns:
            X_["avg_price_by_hour"] = X_.groupby("hour")["price"].transform("mean")

        if "dsp" in X_.columns and "price" in X_.columns:
            X_["price_std_by_dsp"] = X_.groupby("dsp")["price"].transform("std")
            X_["avg_price_by_dsp"] = X_.groupby("dsp")["price"].transform("mean")

        if "request_context_device_h" in X_.columns and "request_context_device_w" in X_.columns:
            # Screen features (replace 0 with NaN)
            X_.loc[X_["request_context_device_h"] == 0, "request_context_device_h"] = np.nan
            X_.loc[X_["request_context_device_w"] == 0, "request_context_device_w"] = np.nan

            X_["screen_ratio"] = X_["request_context_device_w"] / X_["request_context_device_h"]
            X_["screen_diagonal"] = np.sqrt(
                X_["request_context_device_w"] ** 2 + X_["request_context_device_h"] ** 2
            )

        return X_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CalibratedBinaryClassifier":
        """
        Fit the model with feature engineering and calibration.

        Parameters
        ----------
        X : pd.DataFrame
            Training features

        y : pd.Series
            Target variable (binary: 0 or 1)

        Returns
        -------
        self : CalibratedBinaryClassifier
            Fitted model

        Raises
        ------
        ValueError
            If y is not binary or has invalid values

        Notes
        -----
        - Feature engineering is applied automatically
        - Categorical features are detected from X dtypes
        - Calibration is applied after fitting base model
        - Model is calibrated on the same training data (prefit mode)
        """
        # Validate target
        unique_classes = y.unique()
        if len(unique_classes) != 2:
            raise ValueError(
                f"Target must be binary. Found {len(unique_classes)} classes: {unique_classes}"
            )

        # Detect categorical features
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Build and fit base model
        self.model_ = self.build_model(self.variable_params, categorical_features)
        self.model_.fit(X, y)

        # Store metadata
        self.features_ = X.columns.tolist()
        self.classes_ = self.model_.named_steps["classifier"].classes_
        self.feature_importances_ = self.model_.named_steps["classifier"].feature_importances_

        # Apply calibration
        self.calibrated_model_ = MultiCalibrationWrapper(
            base_estimator=self.model_, method=self.calibration_method, **self.calibration_params
        )
        self.calibrated_model_.fit(X, y)

        self.is_fitted_ = True
        return self

    def _optional_feature_extraction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering if needed.

        If input X is missing engineered features, applies feature extraction.
        Otherwise, returns X unchanged.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            Features with engineering applied if needed
        """
        missing_features = set(self.features_) - set(X.columns)
        if len(missing_features) > 0:
            return self.prepare_and_extract_features(X)
        return X

    @check_is_fitted
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Probabilities for [class_0, class_1]

        Examples
        --------
        >>> probas = model.predict_proba(X_test)
        >>> fraud_probabilities = probas[:, 1]
        """
        X = self._optional_feature_extraction(X)
        return self.calibrated_model_.predict_proba(X[self.features_])

    @check_is_fitted
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels (0 or 1)

        Examples
        --------
        >>> predictions = model.predict(X_test)
        >>> accuracy = (predictions == y_test).mean()
        """
        X = self._optional_feature_extraction(X)
        return self.calibrated_model_.predict(X[self.features_])

    @check_is_fitted
    def predict_proba_with_intervals(self, X: pd.DataFrame) -> PredictionIntervals:
        """
        Predict with uncertainty intervals (Venn-ABERS only).

        For Venn-ABERS calibration, returns prediction intervals with
        mathematical guarantees. For other methods, returns point estimates.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on

        Returns
        -------
        dict
            Dictionary with keys:
            - 'p_lower': Lower bound of probability interval (n_samples,)
            - 'p_upper': Upper bound of probability interval (n_samples,)
            - 'p_combined': Combined probability (n_samples,)
            - 'interval_width': Uncertainty measure (n_samples,)
            - 'proba': Standard format probabilities (n_samples, 2)

        Examples
        --------
        >>> intervals = model.predict_proba_with_intervals(X_test)
        >>> print(f"Mean uncertainty: {intervals['interval_width'].mean():.4f}")
        >>>
        >>> # Flag high-uncertainty predictions
        >>> uncertain_mask = intervals['interval_width'] > 0.1
        >>> print(f"Uncertain predictions: {uncertain_mask.sum()} / {len(X_test)}")

        Notes
        -----
        - Venn-ABERS provides [p0, p1] interval with validity guarantees
        - Wide intervals indicate ambiguous/difficult predictions
        - For non-Venn-ABERS methods, interval_width is all zeros
        """
        X = self._optional_feature_extraction(X)
        return self.calibrated_model_.predict_proba_with_intervals(X[self.features_])

    @check_is_fitted
    def calculate_shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SHAP values for model interpretability.

        Parameters
        ----------
        X : pd.DataFrame
            Features to explain

        Returns
        -------
        pd.DataFrame
            SHAP contributions with columns for each feature plus 'expected_value'

        Examples
        --------
        >>> shap_values = model.calculate_shap_values(X_test)
        >>> top_features = shap_values.abs().mean().sort_values(ascending=False).head(10)
        >>> print("Top contributing features:")
        >>> print(top_features)

        Notes
        -----
        - Uses LightGBM's built-in SHAP computation (pred_contrib=True)
        - SHAP values sum to the prediction (expected_value + contributions)
        - Requires feature engineering to be applied first
        """
        X = self._optional_feature_extraction(X)
        X_ = X.copy()

        # Apply all pipeline transformations except classifier
        for step in list(self.model_.named_steps.keys())[:-1]:
            X_ = self.model_.named_steps[step].transform(X_[self.features_])

        # Get SHAP contributions from LightGBM
        contributions = self.model_.named_steps["classifier"].predict_proba(X_, pred_contrib=True)

        contributions_df = pd.DataFrame(
            contributions, columns=X.columns.tolist() + ["expected_value"]
        )

        return contributions_df


# Backward compatibility alias
BidWinModel = CalibratedBinaryClassifier
