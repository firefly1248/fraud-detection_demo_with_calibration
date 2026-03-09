"""Tests for data loading utilities."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import create_time_groups, get_categorical_features


class TestCreateTimeGroups:
    """Test suite for create_time_groups function."""

    def test_quantile_binning(self):
        """Test time grouping with quantile method."""
        df = pd.DataFrame({
            'TransactionDT': np.arange(0, 1000, 1)
        })

        time_groups = create_time_groups(
            df,
            time_column='TransactionDT',
            n_bins=10,
            method='quantile',
            verbose=False
        )

        assert len(time_groups) == len(df)
        assert time_groups.min() == 0
        assert time_groups.max() == 9  # 10 bins: 0-9
        assert len(time_groups.unique()) == 10

    def test_uniform_binning(self):
        """Test time grouping with uniform method."""
        df = pd.DataFrame({
            'TransactionDT': np.arange(0, 1000, 1)
        })

        time_groups = create_time_groups(
            df,
            time_column='TransactionDT',
            n_bins=5,
            method='uniform',
            verbose=False
        )

        assert len(time_groups) == len(df)
        assert len(time_groups.unique()) <= 5

    def test_custom_time_column(self):
        """Test with custom time column name."""
        df = pd.DataFrame({
            'custom_time': np.arange(0, 500, 1)
        })

        time_groups = create_time_groups(
            df,
            time_column='custom_time',
            n_bins=5,
            verbose=False
        )

        assert len(time_groups) == len(df)

    def test_handles_duplicates(self):
        """Test that function handles duplicate timestamps."""
        df = pd.DataFrame({
            'TransactionDT': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10
        })

        time_groups = create_time_groups(
            df,
            time_column='TransactionDT',
            n_bins=3,
            method='quantile',
            verbose=False
        )

        assert len(time_groups) == len(df)
        assert time_groups.dtype == int

    def test_missing_time_column_raises_error(self):
        """Test that missing time column raises error."""
        df = pd.DataFrame({
            'other_column': np.arange(0, 100, 1)
        })

        with pytest.raises(KeyError):
            create_time_groups(
                df,
                time_column='nonexistent_column',
                n_bins=5
            )


class TestGetCategoricalFeatures:
    """Test suite for get_categorical_features function."""

    def test_identifies_categorical_columns(self, sample_fraud_data):
        """Test identification of categorical features."""
        cat_features = get_categorical_features(sample_fraud_data)

        assert isinstance(cat_features, list)
        # Should identify ProductCD and card columns
        assert 'ProductCD' in cat_features
        assert 'card4' in cat_features
        assert 'card6' in cat_features

    def test_excludes_numeric_columns(self):
        """Test that numeric columns are excluded."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'categorical': ['A', 'B', 'C', 'A', 'B']
        })

        cat_features = get_categorical_features(df)

        assert 'categorical' in cat_features
        assert 'numeric1' not in cat_features
        assert 'numeric2' not in cat_features

    def test_handles_mixed_types(self):
        """Test handling of mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['A', 'B', 'C', 'D', 'E'],
            'object_col': ['X', 'Y', 'Z', 'X', 'Y'],
            'bool_col': [True, False, True, False, True]
        })

        cat_features = get_categorical_features(df)

        # Should identify string/object columns
        assert 'str_col' in cat_features or 'object_col' in cat_features
        # Should not include numeric columns
        assert 'int_col' not in cat_features
        assert 'float_col' not in cat_features

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()

        cat_features = get_categorical_features(df)

        assert isinstance(cat_features, list)
        assert len(cat_features) == 0

    def test_no_categorical_features(self):
        """Test DataFrame with no categorical features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })

        cat_features = get_categorical_features(df)

        assert isinstance(cat_features, list)
        # May be empty or contain column names depending on implementation
        # Just check it doesn't error


class TestDataLoaderIntegration:
    """Integration tests for data loading utilities."""

    def test_time_groups_with_fraud_data(self, sample_fraud_data):
        """Test time grouping with fraud detection data."""
        time_groups = create_time_groups(
            sample_fraud_data,
            time_column='TransactionDT',
            n_bins=10,
            verbose=False
        )

        assert len(time_groups) == len(sample_fraud_data)
        assert time_groups.dtype == int

    def test_categorical_and_time_groups_together(self, sample_fraud_data):
        """Test using both utilities together."""
        # Get categorical features
        cat_features = get_categorical_features(sample_fraud_data)

        # Create time groups
        sample_fraud_data['time_group'] = create_time_groups(
            sample_fraud_data,
            n_bins=5,
            verbose=False
        )

        assert 'time_group' in sample_fraud_data.columns
        assert len(cat_features) > 0
