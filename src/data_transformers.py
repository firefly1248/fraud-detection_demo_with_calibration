import typing as tp
from datetime import timedelta

import category_encoders
import numpy as np
import pandas as pd
from feature_engine import imputation
from sklearn.base import BaseEstimator, TransformerMixin

from .config import RANDOM_SEED


class MissingDataHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, cols=None, add_missing_indicator=False, fill_value=-0):
        self.strategy = strategy
        self.cols = cols
        self.add_missing_indicator = add_missing_indicator
        self.fill_value = fill_value
        if strategy == "mean":
            self.imputer = imputation.MeanMedianImputer(imputation_method="mean", variables=cols)
        elif strategy == "median":
            self.imputer = imputation.MeanMedianImputer(imputation_method="median", variables=cols)
        elif strategy == "frequent":
            self.imputer = imputation.CategoricalImputer(
                imputation_method="frequent", variables=cols, ignore_format=True
            )
        elif strategy == "random":
            self.imputer = imputation.RandomSampleImputer(variables=cols)
        elif strategy == "arbitrary":
            self.imputer = imputation.ArbitraryNumberImputer(
                arbitrary_number=fill_value, variables=cols
            )
        elif strategy == "end_of_tail":
            self.imputer = imputation.EndTailImputer(
                imputation_method="max", variables=cols, fold=2
            )
        else:
            raise ValueError("Unknown imputation method")

        if add_missing_indicator:
            self.missing_indicator = imputation.AddMissingIndicator()

    def fit(self, X, y=None):
        if self.add_missing_indicator:
            self.missing_indicator.fit(X)
        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        initial_columns = X.columns
        if self.add_missing_indicator:
            X = self.missing_indicator.transform(X)
        X[initial_columns] = self.imputer.transform(X[initial_columns])
        return X


class EndOfTailImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        if isinstance(self.cols, type(None)):
            self.cols = list(X.select_dtypes(include="number").columns)
        self.imputer_dict_ = (
            X[self.cols].min() - 2 * (X[self.cols].quantile(0.75) - X[self.cols].quantile(0.25))
        ).to_dict()
        return self

    def transform(self, X, y=None):
        return X.fillna(self.imputer_dict_)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(y, type(None)):
            return X.copy()
        return X.copy(), y.copy()

    def inverse_transform(self, X, y=None):
        if isinstance(y, type(None)):
            return X.copy()
        return X.copy(), y.copy()


class FieldsToCategory(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for var in self.variables:
            if var in X.columns:
                X[var] = pd.Categorical(X[var], ordered=False).astype("category")
        return X


class CatFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    Categorical feature encoder with multiple encoding strategies.

    Supports one-hot encoding for low-cardinality features and various
    target-based encodings for high-cardinality features.

    Parameters
    ----------
    strategy : str
        Encoding strategy. Options: 'catboost', 'glmme', 'james_stein',
        'm_estimate', 'polynomial', 'target_encoder', 'backward_difference', 'ordinal'
    handle_missing : str, default='value'
        How to handle missing values in categorical features
    max_cardinality_to_one_hot : int, default=0
        Maximum unique values for one-hot encoding. Features with cardinality
        <= this value will be one-hot encoded, others will use the main strategy
    cols : list of str, optional
        Categorical columns to encode. If None, will auto-detect

    Attributes
    ----------
    ohe : OneHotEncoder or None
        One-hot encoder for low-cardinality features
    handler : Encoder or None
        Main encoder for high-cardinality features
    use_one_hot : bool
        Whether one-hot encoding is used
    use_single_columns_encoding : bool
        Whether main encoding strategy is used
    """

    def __init__(
        self,
        strategy: str,
        handle_missing: str = "value",
        max_cardinality_to_one_hot=0,
        cols: tp.Optional[tp.List[str]] = None,
    ):
        self.strategy = strategy
        self.cols = cols
        self.handle_missing = handle_missing
        self.max_cardinality_to_one_hot = max_cardinality_to_one_hot
        self.use_one_hot = max_cardinality_to_one_hot > 0
        self.use_single_columns_encoding = True
        self.ohe = None
        self.handler = None
        if strategy not in {
            "catboost",
            "glmme",
            "james_stein",
            "m_estimate",
            "polynomial",
            "target_encoder",
            "backward_difference",
            "ordinal",
        }:
            raise ValueError("Unknown stargegy")

    def fit(self, X: pd.DataFrame, y: tp.Optional[pd.Series] = None):
        cols_to_one_hot = [
            c for c in self.cols if len(X[c].unique()) <= self.max_cardinality_to_one_hot
        ]
        cols_to_encode_single_column = [c for c in self.cols if c not in cols_to_one_hot]
        self.use_one_hot = len(cols_to_one_hot) > 0
        self.use_single_columns_encoding = len(cols_to_encode_single_column) > 0
        if self.use_one_hot:
            self.ohe = category_encoders.one_hot.OneHotEncoder(cols=cols_to_one_hot)
            self.ohe.fit(X, y)
        if self.use_single_columns_encoding:
            if self.strategy == "catboost":
                self.handler = category_encoders.cat_boost.CatBoostEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                    random_state=RANDOM_SEED,
                )
            elif self.strategy == "glmme":
                self.handler = category_encoders.glmm.GLMMEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                    random_state=RANDOM_SEED,
                )
            elif self.strategy == "james_stein":
                self.handler = category_encoders.james_stein.JamesSteinEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                    random_state=RANDOM_SEED,
                )
            elif self.strategy == "m_estimate":
                self.handler = category_encoders.m_estimate.MEstimateEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                    random_state=RANDOM_SEED,
                )
            elif self.strategy == "polynomial":
                self.handler = category_encoders.polynomial.PolynomialEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                )
            elif self.strategy == "target_encoder":
                self.handler = category_encoders.target_encoder.TargetEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                )
            elif self.strategy == "backward_difference":
                self.handler = category_encoders.backward_difference.BackwardDifferenceEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                )
            elif self.strategy == "ordinal":
                self.handler = category_encoders.ordinal.OrdinalEncoder(
                    cols=cols_to_encode_single_column,
                    handle_missing=self.handle_missing,
                )
            self.handler.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame, y: tp.Optional[pd.Series] = None):
        X_ = X.copy()
        if self.use_single_columns_encoding:
            X_ = self.handler.transform(X_)
        if self.use_one_hot:
            X_ = self.ohe.transform(X_)
        return X_


class TimeWindowedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder with temporal sliding window to prevent both data leakage and concept drift.

    For each row, computes target encoding statistics using only past data within a specified
    time window [row_timestamp - time_window, row_timestamp). This prevents:
    - **Data leakage**: Only uses past data (like CatBoost encoder)
    - **Concept drift**: Limits history to recent data, ignoring outdated patterns

    The encoder computes smoothed target means for categorical features, where smoothing
    helps with rare categories that have few samples in the time window.

    Parameters
    ----------
    time_column : str
        Name of the timestamp column in the dataset
    time_window : float, int, or timedelta
        Size of the temporal window to look back. Can be:
        - float: interpreted as seconds
        - int: interpreted as days
        - timedelta: explicit time duration
    cols : list of str
        Categorical columns to encode
    smoothing : float, default=1.0
        Smoothing parameter for target encoding. Higher values = more smoothing.
        Formula: (count * target_mean + smoothing * global_mean) / (count + smoothing)
    min_samples_leaf : int, default=1
        Minimum number of samples required in time window to compute category encoding.
        If fewer samples, falls back to global mean
    global_mean : float, optional
        Global target mean to use for smoothing and fallback. If None, computed during fit
    verbose : bool, default=False
        Whether to print progress information during transform

    Attributes
    ----------
    X_train_ : pd.DataFrame
        Training features stored for temporal lookback during transform
    y_train_ : pd.Series
        Training target stored for temporal lookback during transform
    global_mean_ : float
        Global target mean computed from training data
    is_fitted_ : bool
        Whether the encoder has been fitted

    Examples
    --------
    >>> # Encode fraud transactions using only past 30 days of data for each row
    >>> encoder = TimeWindowedTargetEncoder(
    ...     time_column='TransactionDT',
    ...     time_window=30,  # 30 days
    ...     cols=['card1', 'card2', 'ProductCD'],
    ...     smoothing=10.0,
    ...     min_samples_leaf=20
    ... )
    >>> X_encoded = encoder.fit_transform(X_train, y_train)

    Notes
    -----
    - For training data: Uses expanding window (all past data up to current row)
    - For test data: Uses fixed window (time_window before current row)
    - Rows without enough samples in window fall back to global mean
    - Computational complexity: O(n * m) where n=rows, m=avg samples per window
    - For large datasets, consider using parallel processing or caching strategies

    Performance Considerations
    -------------------------
    This encoder can be slow for large datasets since it processes each row individually.
    To improve performance:
    - Use smaller time_window to reduce samples per iteration
    - Increase min_samples_leaf to reduce computation for rare categories
    - Consider sorting data by time_column before fitting
    - Use sample weights or stratification during cross-validation
    """

    def __init__(
        self,
        time_column: str,
        time_window: tp.Union[float, int, timedelta],
        cols: tp.List[str],
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        global_mean: tp.Optional[float] = None,
        verbose: bool = False,
    ):
        self.time_column = time_column
        self.time_window = time_window
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.global_mean = global_mean
        self.verbose = verbose

        # Fitted attributes (trailing underscore per sklearn convention)
        self.X_train_: tp.Optional[pd.DataFrame] = None
        self.y_train_: tp.Optional[pd.Series] = None
        self.global_mean_: float = 0.0
        self.is_fitted_: bool = False

    def _normalize_time_window(self) -> timedelta:
        """
        Convert time_window parameter to timedelta for consistent handling.

        Returns
        -------
        timedelta
            Normalized time window duration
        """
        if isinstance(self.time_window, timedelta):
            return self.time_window
        elif isinstance(self.time_window, int):
            # Interpret as days
            return timedelta(days=self.time_window)
        elif isinstance(self.time_window, float):
            # Interpret as seconds
            return timedelta(seconds=self.time_window)
        else:
            raise ValueError(
                f"time_window must be timedelta, int (days), or float (seconds), "
                f"got {type(self.time_window)}"
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TimeWindowedTargetEncoder":
        """
        Fit the encoder by storing training data and computing global target mean.

        Parameters
        ----------
        X : pd.DataFrame
            Training features including time_column
        y : pd.Series
            Training target (binary or continuous)

        Returns
        -------
        self : TimeWindowedTargetEncoder
            Fitted encoder instance

        Raises
        ------
        ValueError
            If time_column is not in X or if cols contain invalid column names
        """
        if self.time_column not in X.columns:
            raise ValueError(
                f"time_column '{self.time_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        missing_cols = [col for col in self.cols if col not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Categorical columns not found in X: {missing_cols}. "
                f"Available columns: {list(X.columns)}"
            )

        # Store training data for temporal lookback during transform
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Compute global mean for smoothing and fallback
        if self.global_mean is not None:
            self.global_mean_ = self.global_mean
        else:
            self.global_mean_ = float(y.mean())

        self.is_fitted_ = True

        if self.verbose:
            print(f"TimeWindowedTargetEncoder fitted:")
            print(f"  - Training samples: {len(X)}")
            print(f"  - Global target mean: {self.global_mean_:.4f}")
            print(f"  - Time window: {self._normalize_time_window()}")
            print(f"  - Categorical columns: {self.cols}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using time-windowed target encoding.

        For each row, computes target mean for each categorical value using only
        training data from [row_timestamp - time_window, row_timestamp).

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform including time_column

        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame with categorical columns replaced by target-encoded values

        Raises
        ------
        RuntimeError
            If transform is called before fit
        ValueError
            If time_column is not in X
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "TimeWindowedTargetEncoder must be fitted before transform. "
                "Call fit() or fit_transform() first."
            )

        if self.time_column not in X.columns:
            raise ValueError(f"time_column '{self.time_column}' not found in X during transform")

        X_transformed = X.copy()
        time_window_delta = self._normalize_time_window()

        # Convert categorical columns to object dtype to allow float assignment
        # (pandas 3.x infers string columns as StringDtype which rejects float values)
        for col in self.cols:
            X_transformed[col] = X_transformed[col].astype(object)

        # Process each row individually to compute encoding based on its timestamp
        for idx in X_transformed.index:
            current_time = X_transformed.loc[idx, self.time_column]
            window_start = current_time - time_window_delta

            # Filter training data to time window [window_start, current_time)
            mask = (self.X_train_[self.time_column] >= window_start) & (
                self.X_train_[self.time_column] < current_time
            )
            X_window = self.X_train_[mask]
            y_window = self.y_train_[mask]

            # Skip if insufficient samples in window
            if len(X_window) < self.min_samples_leaf:
                for col in self.cols:
                    X_transformed.loc[idx, col] = self.global_mean_
                continue

            # Encode each categorical column
            for col in self.cols:
                cat_value = X_transformed.loc[idx, col]

                # Handle missing values
                if pd.isna(cat_value):
                    X_transformed.loc[idx, col] = self.global_mean_
                    continue

                # Compute target mean for this category in the time window
                mask_category = X_window[col] == cat_value
                y_category = y_window[mask_category]

                if len(y_category) == 0:
                    # Category not seen in window - use global mean
                    encoded_value = self.global_mean_
                else:
                    # Apply smoothing: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
                    count = len(y_category)
                    cat_mean = float(y_category.mean())
                    encoded_value = (count * cat_mean + self.smoothing * self.global_mean_) / (
                        count + self.smoothing
                    )

                X_transformed.loc[idx, col] = encoded_value

        # Cast encoded columns to float so downstream code gets numeric dtype
        for col in self.cols:
            X_transformed[col] = X_transformed[col].astype(float)

        if self.verbose:
            print(f"TimeWindowedTargetEncoder transformed {len(X_transformed)} rows")
            for col in self.cols:
                print(
                    f"  - {col}: mean={X_transformed[col].mean():.4f}, "
                    f"std={X_transformed[col].std():.4f}"
                )

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit encoder and transform training data in one step.

        For training data, uses expanding window approach: each row uses all past data
        up to its timestamp (not limited by time_window) to maximize available information.

        Parameters
        ----------
        X : pd.DataFrame
            Training features including time_column
        y : pd.Series
            Training target

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed training features with target-encoded categorical columns
        """
        self.fit(X, y)

        # For training data, use expanding window (all past data) instead of fixed window
        # This maximizes information while still preventing leakage
        X_transformed = X.copy()

        # Sort by time for efficient expanding window computation
        if not X_transformed[self.time_column].is_monotonic_increasing:
            sort_idx = X_transformed[self.time_column].argsort()
            X_sorted = X_transformed.iloc[sort_idx].reset_index(drop=True)
            y_sorted = y.iloc[sort_idx].reset_index(drop=True)
        else:
            X_sorted = X_transformed.reset_index(drop=True)
            y_sorted = y.reset_index(drop=True)

        # Convert categorical columns to object dtype to allow float assignment
        # (pandas 3.x infers string columns as StringDtype which rejects float values)
        for col in self.cols:
            X_sorted[col] = X_sorted[col].astype(object)

        # Process each row with expanding window
        for idx in range(len(X_sorted)):
            if idx == 0:
                # First row - use global mean
                for col in self.cols:
                    X_sorted.loc[idx, col] = self.global_mean_
                continue

            # Use all past data (expanding window)
            X_past = X_sorted.iloc[:idx]
            y_past = y_sorted.iloc[:idx]

            # Encode each categorical column
            for col in self.cols:
                cat_value = X_sorted.loc[idx, col]

                if pd.isna(cat_value):
                    X_sorted.loc[idx, col] = self.global_mean_
                    continue

                # Compute target mean for this category using all past data
                mask_category = X_past[col] == cat_value
                y_category = y_past[mask_category]

                if len(y_category) == 0:
                    encoded_value = self.global_mean_
                else:
                    count = len(y_category)
                    cat_mean = float(y_category.mean())
                    encoded_value = (count * cat_mean + self.smoothing * self.global_mean_) / (
                        count + self.smoothing
                    )

                X_sorted.loc[idx, col] = encoded_value

        # Restore original order if data was sorted
        if not X_transformed[self.time_column].is_monotonic_increasing:
            X_sorted = X_sorted.iloc[sort_idx.argsort()].reset_index(drop=True)

        # Cast encoded columns to float so downstream code gets numeric dtype
        for col in self.cols:
            X_sorted[col] = X_sorted[col].astype(float)

        if self.verbose:
            print(f"TimeWindowedTargetEncoder fit_transform completed for {len(X_sorted)} rows")

        return X_sorted
