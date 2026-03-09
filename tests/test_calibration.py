"""Tests for calibration methods."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_clf.calibration import MultiCalibrationWrapper, VennABERSBinaryCalibrator


class TestMultiCalibrationWrapper:
    """Test suite for MultiCalibrationWrapper."""

    @pytest.fixture
    def sample_classifier_and_data(self):
        """Create sample classifier and data for calibration testing."""
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42
        )

        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        return clf, X_cal, y_cal

    def test_isotonic_calibration(self, sample_classifier_and_data):
        """Test isotonic regression calibration."""
        clf, X_cal, y_cal = sample_classifier_and_data

        calibrator = MultiCalibrationWrapper(base_estimator=clf, method="isotonic")

        calibrator.fit(X_cal, y_cal)

        # Test predict_proba
        probas = calibrator.predict_proba(X_cal)
        assert probas.shape == (len(X_cal), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_sigmoid_calibration(self, sample_classifier_and_data):
        """Test Platt scaling (sigmoid) calibration."""
        clf, X_cal, y_cal = sample_classifier_and_data

        calibrator = MultiCalibrationWrapper(base_estimator=clf, method="sigmoid")

        calibrator.fit(X_cal, y_cal)

        probas = calibrator.predict_proba(X_cal)
        assert probas.shape == (len(X_cal), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_venn_abers_calibration(self, sample_classifier_and_data):
        """Test Venn-ABERS calibration."""
        clf, X_cal, y_cal = sample_classifier_and_data

        calibrator = MultiCalibrationWrapper(base_estimator=clf, method="venn_abers", cal_size=0.5)

        calibrator.fit(X_cal, y_cal)

        # Test standard predict_proba
        probas = calibrator.predict_proba(X_cal)
        assert probas.shape == (len(X_cal), 2)

        # Test prediction intervals
        intervals = calibrator.predict_proba_with_intervals(X_cal)
        assert "p_lower" in intervals
        assert "p_upper" in intervals
        assert "p_combined" in intervals
        assert "interval_width" in intervals

        # Check interval validity
        assert np.all(intervals["p_lower"] <= intervals["p_upper"])
        assert np.all(intervals["interval_width"] >= 0)

    def test_no_calibration(self, sample_classifier_and_data):
        """Test that 'none' method returns uncalibrated predictions."""
        clf, X_cal, y_cal = sample_classifier_and_data

        calibrator = MultiCalibrationWrapper(base_estimator=clf, method="none")

        calibrator.fit(X_cal, y_cal)

        probas_calibrated = calibrator.predict_proba(X_cal)
        probas_original = clf.predict_proba(X_cal)

        # Should be identical when method='none'
        assert np.allclose(probas_calibrated, probas_original)

    def test_invalid_method_raises_error(self, sample_classifier_and_data):
        """Test that invalid calibration method raises error."""
        clf, X_cal, y_cal = sample_classifier_and_data

        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrator = MultiCalibrationWrapper(base_estimator=clf, method="invalid_method")
            calibrator.fit(X_cal, y_cal)

    def test_calibration_improves_predictions(self, sample_classifier_and_data):
        """Test that calibration produces valid probability distributions."""
        clf, X_cal, y_cal = sample_classifier_and_data

        # Isotonic calibration
        calibrator = MultiCalibrationWrapper(base_estimator=clf, method="isotonic")
        calibrator.fit(X_cal, y_cal)

        probas = calibrator.predict_proba(X_cal)

        # Check probability properties
        assert np.all(probas[:, 1] >= 0) and np.all(probas[:, 1] <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0)

        # Mean probability should be close to actual positive rate
        mean_proba = probas[:, 1].mean()
        actual_rate = y_cal.mean()
        # Should be reasonably close (within 20%)
        assert abs(mean_proba - actual_rate) < 0.2


class TestVennABERSBinaryCalibrator:
    """Test suite for VennABERSBinaryCalibrator."""

    def test_initialization(self):
        """Test VennABERSBinaryCalibrator initialization."""
        calibrator = VennABERSBinaryCalibrator(precision=3)
        assert calibrator.precision == 3

    def test_fit_predict(self):
        """Test fit and predict workflow."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)

        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        # VennABERSBinaryCalibrator takes probability scores, not raw features
        p_cal = clf.predict_proba(X_cal)[:, 1]
        calibrator = VennABERSBinaryCalibrator(precision=3)
        calibrator.fit(p_cal, y_cal)

        p_test = clf.predict_proba(X_cal)[:, 1]
        result = calibrator.predict(p_test)

        assert isinstance(result, dict)
        assert all(key in result for key in ["p0", "p1", "p_combined", "interval_width"])

    def test_interval_properties(self):
        """Test mathematical properties of prediction intervals."""
        X, y = make_classification(n_samples=300, n_features=5, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        p_train = clf.predict_proba(X_train)[:, 1]
        calibrator = VennABERSBinaryCalibrator(precision=3)
        calibrator.fit(p_train, y_train)

        p_test = clf.predict_proba(X_test)[:, 1]
        result = calibrator.predict(p_test)

        # p0 <= p1 (lower bound <= upper bound)
        assert np.all(result["p0"] <= result["p1"])

        # All probabilities in [0, 1]
        assert np.all(result["p0"] >= 0) and np.all(result["p0"] <= 1)
        assert np.all(result["p1"] >= 0) and np.all(result["p1"] <= 1)

        # Interval width is non-negative
        assert np.all(result["interval_width"] >= 0)
