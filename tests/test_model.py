"""Tests for CalibratedBinaryClassifier."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.model import CalibratedBinaryClassifier, BidWinModel


class TestCalibratedBinaryClassifier:
    """Test suite for CalibratedBinaryClassifier."""

    def test_initialization(self, model_params):
        """Test model initialization with various calibration methods."""
        # Test isotonic calibration (default)
        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='isotonic'
        )
        assert model.calibration_method == 'isotonic'
        assert not model.is_fitted_

        # Test venn_abers calibration
        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='venn_abers',
            calibration_params={'cal_size': 0.2}
        )
        assert model.calibration_method == 'venn_abers'

        # Test no calibration
        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='none'
        )
        assert model.calibration_method == 'none'

    def test_fit_predict(self, sample_binary_data, model_params):
        """Test basic fit and predict workflow."""
        X, y = sample_binary_data

        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='isotonic'
        )

        # Test fit
        model.fit(X, y)
        assert model.is_fitted_
        assert model.model_ is not None

        # Test predict
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

        # Test predict_proba
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_predict_before_fit_raises_error(self, sample_binary_data, model_params):
        """Test that predict raises error before fit."""
        X, y = sample_binary_data

        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='isotonic'
        )

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)

    def test_venn_abers_intervals(self, sample_binary_data, model_params):
        """Test Venn-ABERS prediction intervals."""
        X, y = sample_binary_data

        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='venn_abers',
            calibration_params={'cal_size': 0.2}
        )

        model.fit(X, y)

        # Test prediction intervals
        intervals = model.predict_proba_with_intervals(X)

        assert 'p_lower' in intervals
        assert 'p_upper' in intervals
        assert 'p_combined' in intervals
        assert 'interval_width' in intervals

        # Check interval properties
        assert len(intervals['p_lower']) == len(X)
        assert np.all(intervals['p_lower'] <= intervals['p_upper'])
        assert np.all(intervals['interval_width'] >= 0)

    def test_feature_engineering_fraud(self, sample_fraud_data):
        """Test automatic feature engineering for fraud data."""
        X = sample_fraud_data.drop(columns=['isFraud'])

        X_eng = CalibratedBinaryClassifier.prepare_and_extract_features(X)

        # Check that new features were created
        assert 'TransactionAmt_log' in X_eng.columns
        assert 'TransactionDT_hour' in X_eng.columns
        assert X_eng.shape[1] > X.shape[1]  # More features after engineering

    def test_backward_compatibility_alias(self, sample_binary_data, model_params):
        """Test that BidWinModel alias works correctly."""
        X, y = sample_binary_data

        # Should work identically to CalibratedBinaryClassifier
        model = BidWinModel(
            variable_params=model_params,
            calibration_method='isotonic'
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert isinstance(model, CalibratedBinaryClassifier)

    def test_different_calibration_methods(self, sample_binary_data, model_params):
        """Test all calibration methods produce valid outputs."""
        X, y = sample_binary_data

        for method in ['isotonic', 'sigmoid', 'none']:
            model = CalibratedBinaryClassifier(
                variable_params=model_params,
                calibration_method=method
            )

            model.fit(X, y)
            probas = model.predict_proba(X)

            assert probas.shape == (len(X), 2)
            assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_binary_target_validation(self, sample_binary_data, model_params):
        """Test that model validates binary target."""
        X, y = sample_binary_data

        # Modify y to have 3 classes
        y_multiclass = pd.Series([0, 1, 2] * (len(y) // 3))

        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='isotonic'
        )

        with pytest.raises(ValueError, match="binary"):
            model.fit(X, y_multiclass)

    def test_model_persistence_attributes(self, sample_binary_data, model_params):
        """Test that fitted attributes follow sklearn conventions."""
        X, y = sample_binary_data

        model = CalibratedBinaryClassifier(
            variable_params=model_params,
            calibration_method='isotonic'
        )

        # Before fit
        assert not model.is_fitted_
        assert model.model_ is None

        # After fit
        model.fit(X, y)
        assert model.is_fitted_
        assert model.model_ is not None
        assert hasattr(model, 'features_')
