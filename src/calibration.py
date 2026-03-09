"""
Advanced Calibration Methods for Binary Classification

Implements and compares multiple calibration approaches:
1. Isotonic Regression (sklearn standard)
2. Venn-ABERS Conformal Prediction (provides prediction intervals)
3. Platt Scaling (for comparison)
"""

import typing as tp
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from venn_abers import VennAbersCalibrator


class DataFrameWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper that converts numpy arrays to DataFrames before passing to wrapped estimator.

    This is needed because VennAbersCalibrator internally converts data to numpy arrays,
    but our pipeline expects DataFrames with column names for categorical encoding.
    """

    def __init__(self, estimator: BaseEstimator, feature_names: tp.List[str]):
        self.estimator = estimator
        self.feature_names = feature_names

    def _ensure_dataframe(self, X):
        """Convert numpy array to DataFrame if needed."""
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names)
        return X

    def fit(self, X, y):
        """Fit wrapped estimator with DataFrame conversion."""
        X_df = self._ensure_dataframe(X)
        self.estimator.fit(X_df, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        """Predict with DataFrame conversion."""
        X_df = self._ensure_dataframe(X)
        return self.estimator.predict(X_df)

    def predict_proba(self, X):
        """Predict probabilities with DataFrame conversion."""
        X_df = self._ensure_dataframe(X)
        return self.estimator.predict_proba(X_df)


class MultiCalibrationWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper that supports multiple calibration methods for comparison.

    Calibration methods:
    - 'isotonic': Standard isotonic regression (sklearn)
    - 'venn_abers': Venn-ABERS conformal prediction (provides intervals)
    - 'sigmoid': Platt scaling (logistic regression)
    - None: No calibration

    For Venn-ABERS, returns prediction intervals [p0, p1] with mathematical
    guarantees that true probability lies within the interval.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        method: str = "isotonic",
        venn_abers_mode: str = "inductive",
        cal_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 0,
    ):
        """
        Args:
            base_estimator: Fitted classifier to calibrate
            method: Calibration method ('isotonic', 'venn_abers', 'sigmoid', None)
            venn_abers_mode: 'inductive' (faster) or 'cross' (more robust)
            cal_size: Calibration set size for inductive Venn-ABERS (0.0-1.0)
            cv_folds: Number of folds for cross Venn-ABERS
            random_state: Random seed
        """
        self.base_estimator = base_estimator
        self.method = method
        self.venn_abers_mode = venn_abers_mode
        self.cal_size = cal_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.calibrator_ = None
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit calibration on top of base estimator."""
        self.classes_ = np.unique(y)

        if self.method is None or self.method == "none":
            # No calibration
            self.calibrator_ = None
            return self

        elif self.method in ("isotonic", "sigmoid"):
            # Split off a held-out calibration set so the calibrator is never
            # fitted on the same data the base model was trained on.
            # Training on the same data causes the isotonic/sigmoid layer to
            # overfit to the overconfident in-sample predictions of LightGBM,
            # which degrades calibration quality on the test set.
            from sklearn.model_selection import train_test_split
            from sklearn.base import clone

            X_proper, X_cal, y_proper, y_cal = train_test_split(
                X, y, test_size=self.cal_size, random_state=self.random_state, stratify=y
            )

            # Refit the base model on the proper training set only
            self.base_estimator = clone(self.base_estimator)
            self.base_estimator.fit(X_proper, y_proper)

            # Fit the calibration layer on the held-out set
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_estimator, method=self.method, cv="prefit"
            )
            self.calibrator_.fit(X_cal, y_cal)

        elif self.method == "venn_abers":
            # Venn-ABERS conformal prediction using rigorous implementation
            # This avoids the issue of VennAbersCalibrator refitting the pipeline
            # with numpy arrays when it expects DataFrames
            from sklearn.model_selection import train_test_split

            # For inductive Venn-ABERS, split into proper training and calibration sets
            if self.venn_abers_mode == "inductive":
                # Split data for inductive calibration
                X_proper, X_cal, y_proper, y_cal = train_test_split(
                    X, y, test_size=self.cal_size, random_state=self.random_state, stratify=y
                )

                # Refit base model on proper training set only
                from sklearn.base import clone

                self.base_estimator = clone(self.base_estimator)
                self.base_estimator.fit(X_proper, y_proper)

                # Get predictions on calibration set
                p_cal = self.base_estimator.predict_proba(X_cal)[:, 1]

                # Fit RIGOROUS Venn-ABERS calibrator with conformal guarantees
                # Note: This is slower but mathematically correct
                self.calibrator_ = RigorousVennABERSCalibrator(
                    precision=3,  # Round to 3 decimals for speed
                    use_cache=True,  # Cache results for repeated values
                )
                self.calibrator_.fit(p_cal, y_cal.values if hasattr(y_cal, "values") else y_cal)
            else:
                # For cross-validation mode, use all data
                # (simplified: could implement k-fold cross-calibration)
                p_cal = self.base_estimator.predict_proba(X)[:, 1]
                self.calibrator_ = RigorousVennABERSCalibrator(precision=3, use_cache=True)
                self.calibrator_.fit(p_cal, y.values if hasattr(y, "values") else y)

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Returns:
            For isotonic/sigmoid: (n_samples, 2) array of probabilities
            For venn_abers: (n_samples, 2) array using combined probability
        """
        if self.calibrator_ is None:
            # No calibration - use base estimator
            return self.base_estimator.predict_proba(X)

        if self.method == "venn_abers":
            # For manual Venn-ABERS, first get predictions from base estimator
            p_test = self.base_estimator.predict_proba(X)[:, 1]

            # Then calibrate them
            va_output = self.calibrator_.predict(p_test)
            p1_combined = va_output["p_combined"]

            # Return in sklearn format: [P(class=0), P(class=1)]
            p0_combined = 1 - p1_combined
            return np.column_stack([p0_combined, p1_combined])
        else:
            # Standard calibration (isotonic/sigmoid)
            return self.calibrator_.predict_proba(X)

    def predict_proba_with_intervals(self, X: pd.DataFrame) -> tp.Dict[str, np.ndarray]:
        """
        Predict with prediction intervals (only for Venn-ABERS).

        Returns:
            Dictionary with:
            - 'p_lower': Lower bound of probability interval (p0)
            - 'p_upper': Upper bound of probability interval (p1)
            - 'p_combined': Combined probability p1/(1-p0+p1)
            - 'interval_width': Uncertainty measure (p1 - p0)
            - 'proba': Standard format [P(class=0), P(class=1)]

        For non-Venn-ABERS methods, returns point estimates with zero width.
        """
        if self.method != "venn_abers":
            # For non-Venn-ABERS, return point estimates
            proba = self.predict_proba(X)
            p1 = proba[:, 1]
            return {
                "p_lower": p1,
                "p_upper": p1,
                "p_combined": p1,
                "interval_width": np.zeros_like(p1),
                "proba": proba,
            }

        # For manual Venn-ABERS, first get predictions from base estimator
        p_test = self.base_estimator.predict_proba(X)[:, 1]

        # Then get calibrated intervals
        va_output = self.calibrator_.predict(p_test)

        p0 = va_output["p0"]
        p1 = va_output["p1"]
        p_combined = va_output["p_combined"]
        interval_width = va_output["interval_width"]

        # Create full probability array [P(class=0), P(class=1)]
        proba_full = np.column_stack([1 - p_combined, p_combined])

        return {
            "p_lower": p0,
            "p_upper": p1,
            "p_combined": p_combined,
            "interval_width": interval_width,
            "proba": proba_full,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class VennABERSBinaryCalibrator(BaseEstimator):
    """
    Simplified Venn-ABERS calibrator (fast but not conformally valid).

    ⚠️ WARNING: This is an approximation that adds boundary points [0,0] and [1,1]
    once during fit, rather than adding each test point individually. This is much
    faster but does NOT provide the conformal prediction guarantees of proper
    Venn-ABERS. Use RigorousVennABERSCalibrator for scientifically correct implementation.
    """

    def __init__(self, precision: int = None):
        """
        Args:
            precision: Number of decimal places for rounding (speeds up computation)
        """
        self.precision = precision
        self.iso_regressor_0_ = None
        self.iso_regressor_1_ = None

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray):
        """
        Fit two isotonic regressors on calibration data.

        Args:
            p_cal: Uncalibrated probabilities from base model (n_samples,)
            y_cal: True labels (n_samples,)
        """
        p_cal = np.asarray(p_cal).flatten()
        y_cal = np.asarray(y_cal).flatten()

        if self.precision is not None:
            p_cal = np.round(p_cal, self.precision)

        # Isotonic regressor assuming all test points are class 0
        p_cal_0 = np.concatenate([p_cal, [0]])
        y_cal_0 = np.concatenate([y_cal, [0]])
        self.iso_regressor_0_ = IsotonicRegression(out_of_bounds="clip")
        self.iso_regressor_0_.fit(p_cal_0, y_cal_0)

        # Isotonic regressor assuming all test points are class 1
        p_cal_1 = np.concatenate([p_cal, [1]])
        y_cal_1 = np.concatenate([y_cal, [1]])
        self.iso_regressor_1_ = IsotonicRegression(out_of_bounds="clip")
        self.iso_regressor_1_.fit(p_cal_1, y_cal_1)

        return self

    def predict(self, p_test: np.ndarray) -> tp.Dict[str, np.ndarray]:
        """
        Predict calibrated probabilities with intervals.

        Args:
            p_test: Uncalibrated probabilities from base model

        Returns:
            Dictionary with:
            - 'p0': Lower bound (assuming test object is class 0)
            - 'p1': Upper bound (assuming test object is class 1)
            - 'p_combined': Combined probability p1/(1-p0+p1)
            - 'interval_width': Uncertainty measure (p1 - p0)
        """
        p_test = np.asarray(p_test).flatten()

        if self.precision is not None:
            p_test = np.round(p_test, self.precision)

        # Get calibrated probabilities from both isotonic regressors
        p0 = self.iso_regressor_0_.predict(p_test)
        p1 = self.iso_regressor_1_.predict(p_test)

        # Combined probability (recommended for decision making)
        p_combined = p1 / (1 - p0 + p1 + 1e-10)  # Add epsilon for numerical stability

        # Interval width (uncertainty measure)
        interval_width = p1 - p0

        return {"p0": p0, "p1": p1, "p_combined": p_combined, "interval_width": interval_width}


class RigorousVennABERSCalibrator(BaseEstimator):
    """
    Rigorous Venn-ABERS calibrator with conformal prediction guarantees.

    This implements the proper Venn-ABERS algorithm from the paper:
    "Large-scale probabilistic predictors with and without guarantees of validity"
    by Vovk et al.

    For each test point, it adds that point to the calibration set with both
    possible labels (0 and 1), fits isotonic regression, and returns the
    prediction intervals [p0, p1] with validity guarantees.

    ⚠️ Note: This is O(n_test × n_cal) complexity, so slower than the simplified
    version, but provides mathematically rigorous conformal prediction intervals.

    Parameters
    ----------
    precision : int, optional
        Number of decimal places for rounding predictions (speeds up computation
        by reducing unique values). None means no rounding.

    use_cache : bool, default=True
        Cache isotonic regression results for identical prediction values.
        Significantly speeds up computation when many predictions have the same value.
    """

    def __init__(self, precision: tp.Optional[int] = None, use_cache: bool = True):
        self.precision = precision
        self.use_cache = use_cache
        self.p_cal_ = None
        self.y_cal_ = None
        self._cache_0 = {}  # Cache for p0 predictions
        self._cache_1 = {}  # Cache for p1 predictions

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray):
        """
        Store calibration data for use during prediction.

        Args:
            p_cal: Uncalibrated probabilities from base model (n_samples,)
            y_cal: True labels (n_samples,)
        """
        self.p_cal_ = np.asarray(p_cal).flatten()
        self.y_cal_ = np.asarray(y_cal).flatten()

        if self.precision is not None:
            self.p_cal_ = np.round(self.p_cal_, self.precision)

        # Clear cache
        self._cache_0 = {}
        self._cache_1 = {}

        return self

    def predict(self, p_test: np.ndarray) -> tp.Dict[str, np.ndarray]:
        """
        Predict calibrated probabilities with conformal intervals.

        For each test prediction p_t:
        1. Add (p_t, 0) to calibration set, fit isotonic regression → get p0
        2. Add (p_t, 1) to calibration set, fit isotonic regression → get p1
        3. Return interval [p0, p1] and combined probability

        Args:
            p_test: Uncalibrated probabilities from base model

        Returns:
            Dictionary with:
            - 'p0': Lower bound (probability assuming test object is class 0)
            - 'p1': Upper bound (probability assuming test object is class 1)
            - 'p_combined': Combined probability p1/(1-p0+p1)
            - 'interval_width': Uncertainty measure (p1 - p0)
        """
        p_test = np.asarray(p_test).flatten()

        if self.precision is not None:
            p_test = np.round(p_test, self.precision)

        n_test = len(p_test)
        p0 = np.zeros(n_test)
        p1 = np.zeros(n_test)

        # For each test point, fit isotonic regression with that point added
        for i, p_t in enumerate(p_test):
            # Check cache first if enabled
            cache_key = float(p_t) if self.use_cache else None

            # p0: Add (p_t, 0) to calibration set
            if self.use_cache and cache_key in self._cache_0:
                p0[i] = self._cache_0[cache_key]
            else:
                p_cal_0 = np.concatenate([self.p_cal_, [p_t]])
                y_cal_0 = np.concatenate([self.y_cal_, [0]])
                iso_0 = IsotonicRegression(out_of_bounds="clip")
                iso_0.fit(p_cal_0, y_cal_0)
                p0[i] = iso_0.predict([p_t])[0]

                if self.use_cache:
                    self._cache_0[cache_key] = p0[i]

            # p1: Add (p_t, 1) to calibration set
            if self.use_cache and cache_key in self._cache_1:
                p1[i] = self._cache_1[cache_key]
            else:
                p_cal_1 = np.concatenate([self.p_cal_, [p_t]])
                y_cal_1 = np.concatenate([self.y_cal_, [1]])
                iso_1 = IsotonicRegression(out_of_bounds="clip")
                iso_1.fit(p_cal_1, y_cal_1)
                p1[i] = iso_1.predict([p_t])[0]

                if self.use_cache:
                    self._cache_1[cache_key] = p1[i]

        # Combined probability (recommended for decision making)
        # This is the Venn-ABERS combination formula
        epsilon = 1e-10
        p_combined = p1 / (1 - p0 + p1 + epsilon)

        # Clip to valid probability range
        p_combined = np.clip(p_combined, 0, 1)

        # Interval width (uncertainty measure)
        interval_width = np.maximum(p1 - p0, 0)

        return {"p0": p0, "p1": p1, "p_combined": p_combined, "interval_width": interval_width}


def compare_calibration_methods(
    model: BaseEstimator,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    methods: tp.List[str] = ["isotonic", "venn_abers", "sigmoid"],
) -> pd.DataFrame:
    """
    Compare multiple calibration methods on the same data.

    Args:
        model: Fitted base model
        X_cal: Calibration features
        y_cal: Calibration labels
        X_test: Test features
        y_test: Test labels
        methods: List of calibration methods to compare

    Returns:
        DataFrame with comparison metrics for each method
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.calibration import calibration_curve

    results = []

    for method in methods:
        # Fit calibrator
        calibrator = MultiCalibrationWrapper(
            base_estimator=model, method=method, cal_size=0.2, random_state=0
        )
        calibrator.fit(X_cal, y_cal)

        # Predict on test set
        y_pred_proba = calibrator.predict_proba(X_test)[:, 1]

        # Compute metrics
        brier = brier_score_loss(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Calibration curve (Expected Calibration Error)
        prob_true, prob_pred = calibration_curve(
            y_test, y_pred_proba, n_bins=10, strategy="quantile"
        )
        ece = np.mean(np.abs(prob_true - prob_pred))

        # For Venn-ABERS, also track interval width
        interval_info = calibrator.predict_proba_with_intervals(X_test)
        avg_interval_width = interval_info["interval_width"].mean()

        results.append(
            {
                "method": method,
                "brier_score": brier,
                "log_loss": logloss,
                "auc_roc": auc_roc,
                "ece": ece,
                "avg_interval_width": avg_interval_width,
            }
        )

    return pd.DataFrame(results)
