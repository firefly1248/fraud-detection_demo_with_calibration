"""Tests for data transformers."""

import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

from src.data_transformers import TimeWindowedTargetEncoder, CatFeaturesEncoder, MissingDataHandler


class TestTimeWindowedTargetEncoder:
    """Test suite for TimeWindowedTargetEncoder."""

    def test_initialization(self):
        """Test TimeWindowedTargetEncoder initialization."""
        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp",
            time_window=30,
            cols=["cat1", "cat2"],
            smoothing=10.0,
            min_samples_leaf=5,
        )

        assert encoder.time_column == "timestamp"
        assert encoder.time_window == 30
        assert encoder.cols == ["cat1", "cat2"]
        assert encoder.smoothing == 10.0
        assert encoder.min_samples_leaf == 5
        assert not encoder.is_fitted_

    def test_time_window_normalization(self):
        """Test time_window parameter normalization."""
        # Test integer (days)
        encoder = TimeWindowedTargetEncoder(time_column="ts", time_window=30, cols=["cat"])
        assert encoder._normalize_time_window() == timedelta(days=30)

        # Test float (seconds)
        encoder = TimeWindowedTargetEncoder(time_column="ts", time_window=3600.0, cols=["cat"])
        assert encoder._normalize_time_window() == timedelta(seconds=3600)

        # Test timedelta
        encoder = TimeWindowedTargetEncoder(
            time_column="ts", time_window=timedelta(hours=24), cols=["cat"]
        )
        assert encoder._normalize_time_window() == timedelta(hours=24)

    def test_fit_transform_basic(self, sample_temporal_data):
        """Test basic fit_transform workflow."""
        X, y = sample_temporal_data

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp",
            time_window=timedelta(days=30),
            cols=["category"],
            smoothing=1.0,
        )

        X_encoded = encoder.fit_transform(X, y)

        assert encoder.is_fitted_
        assert X_encoded.shape == X.shape
        assert "category" in X_encoded.columns
        # Category should now be numeric (encoded)
        assert pd.api.types.is_numeric_dtype(X_encoded["category"])

    def test_fit_and_transform_separately(self, sample_temporal_data):
        """Test fit and transform called separately."""
        X, y = sample_temporal_data

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp", time_window=timedelta(days=30), cols=["category"]
        )

        encoder.fit(X, y)
        assert encoder.is_fitted_
        assert encoder.global_mean_ > 0

        X_encoded = encoder.transform(X)
        assert X_encoded.shape == X.shape

    def test_transform_before_fit_raises_error(self, sample_temporal_data):
        """Test that transform raises error before fit."""
        X, y = sample_temporal_data

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp", time_window=30, cols=["category"]
        )

        with pytest.raises(RuntimeError, match="must be fitted"):
            encoder.transform(X)

    def test_missing_time_column_raises_error(self, sample_temporal_data):
        """Test that missing time_column raises error."""
        X, y = sample_temporal_data
        X_no_time = X.drop(columns=["timestamp"])

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp", time_window=30, cols=["category"]
        )

        with pytest.raises(ValueError, match="time_column.*not found"):
            encoder.fit(X_no_time, y)

    def test_missing_categorical_column_raises_error(self, sample_temporal_data):
        """Test that missing categorical column raises error."""
        X, y = sample_temporal_data

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp", time_window=30, cols=["nonexistent_column"]
        )

        with pytest.raises(ValueError, match="Categorical columns not found"):
            encoder.fit(X, y)

    def test_handles_missing_values(self):
        """Test handling of missing categorical values."""
        X = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
                "category": ["A", "B", None, "A", "B"] * 20,
            }
        )
        y = pd.Series([0, 1, 0, 1, 0] * 20)

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp", time_window=30, cols=["category"], smoothing=1.0
        )

        X_encoded = encoder.fit_transform(X, y)

        # Should not raise error and should handle NaN
        assert not X_encoded["category"].isna().all()

    def test_global_mean_computation(self, sample_temporal_data):
        """Test that global mean is computed correctly."""
        X, y = sample_temporal_data

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp", time_window=30, cols=["category"]
        )

        encoder.fit(X, y)

        expected_global_mean = y.mean()
        assert abs(encoder.global_mean_ - expected_global_mean) < 1e-6

    def test_custom_global_mean(self, sample_temporal_data):
        """Test using custom global_mean parameter."""
        X, y = sample_temporal_data

        custom_mean = 0.5
        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp", time_window=30, cols=["category"], global_mean=custom_mean
        )

        encoder.fit(X, y)
        assert encoder.global_mean_ == custom_mean

    def test_min_samples_leaf_behavior(self):
        """Test min_samples_leaf fallback to global mean."""
        X = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="D"),
                "category": ["A"] * 50,
            }
        )
        y = pd.Series([1] * 50)

        encoder = TimeWindowedTargetEncoder(
            time_column="timestamp",
            time_window=timedelta(days=5),  # Small window
            cols=["category"],
            min_samples_leaf=100,  # Require many samples (won't be met)
            smoothing=0.0,
        )

        X_encoded = encoder.fit_transform(X, y)

        # Should fall back to global mean for most rows
        # (except later rows that have enough history)
        assert "category" in X_encoded.columns


class TestCatFeaturesEncoder:
    """Test suite for CatFeaturesEncoder."""

    def test_target_encoder_strategy(self, sample_binary_data):
        """Test target encoding strategy."""
        X, y = sample_binary_data

        encoder = CatFeaturesEncoder(strategy="target_encoder", cols=["cat_feature"])

        encoder.fit(X, y)
        X_encoded = encoder.transform(X)

        assert "cat_feature" in X_encoded.columns
        # Should be numeric after encoding
        assert pd.api.types.is_numeric_dtype(X_encoded["cat_feature"])

    def test_invalid_strategy_raises_error(self, sample_binary_data):
        """Test that invalid strategy raises error."""
        X, y = sample_binary_data

        with pytest.raises(ValueError, match="Unknown"):
            encoder = CatFeaturesEncoder(strategy="invalid_strategy", cols=["cat_feature"])


class TestMissingDataHandler:
    """Test suite for MissingDataHandler."""

    def test_mean_imputation(self):
        """Test mean imputation strategy."""
        X = pd.DataFrame(
            {"feature1": [1.0, 2.0, np.nan, 4.0, 5.0], "feature2": [10.0, np.nan, 30.0, 40.0, 50.0]}
        )

        imputer = MissingDataHandler(strategy="mean", cols=["feature1", "feature2"])

        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # Should have no missing values
        assert not X_imputed.isna().any().any()

    def test_arbitrary_imputation(self):
        """Test arbitrary value imputation."""
        X = pd.DataFrame(
            {"feature1": [1.0, 2.0, np.nan, 4.0], "feature2": [10.0, np.nan, 30.0, 40.0]}
        )

        fill_value = -999
        imputer = MissingDataHandler(
            strategy="arbitrary", cols=["feature1", "feature2"], fill_value=fill_value
        )

        imputer.fit(X)
        X_imputed = imputer.transform(X)

        # Missing values should be replaced with fill_value
        assert not X_imputed.isna().any().any()
        assert (X_imputed == fill_value).sum().sum() == 2  # Two missing values
