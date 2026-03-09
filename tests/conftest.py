"""Pytest configuration and fixtures for testing."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_binary_data():
    """
    Create sample binary classification dataset for testing.

    Returns
    -------
    tuple
        (X, y) where X is features DataFrame and y is binary target Series
    """
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "cat_feature": np.random.choice(["A", "B", "C", "D"], n_samples),
        }
    )

    # Create target with some signal
    y = pd.Series((X["feature1"] + X["feature2"] > 0).astype(int), name="target")

    return X, y


@pytest.fixture
def sample_fraud_data():
    """
    Create sample fraud detection dataset mimicking IEEE format.

    Returns
    -------
    pd.DataFrame
        DataFrame with fraud-like features including TransactionDT
    """
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame(
        {
            "TransactionID": range(1, n_samples + 1),
            "TransactionDT": np.arange(0, n_samples * 3600, 3600),  # Hourly transactions
            "TransactionAmt": np.random.exponential(100, n_samples),
            "ProductCD": np.random.choice(["W", "C", "H", "R"], n_samples),
            "card1": np.random.randint(1000, 20000, n_samples),
            "card2": np.random.randint(100, 500, n_samples),
            "card4": np.random.choice(["visa", "mastercard", "discover"], n_samples),
            "card6": np.random.choice(["credit", "debit"], n_samples),
            "P_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", "outlook.com", None], n_samples
            ),
            "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com", None], n_samples),
            "addr1": np.random.randint(100, 500, n_samples),
            "addr2": np.random.randint(10, 100, n_samples),
            "dist1": np.random.exponential(10, n_samples),
            "D1": np.random.exponential(5, n_samples),
            "D2": np.random.exponential(5, n_samples),
            "M1": np.random.choice(["T", "F", None], n_samples),
            "M2": np.random.choice(["T", "F", None], n_samples),
            "isFraud": np.random.choice([0, 1], n_samples, p=[0.965, 0.035]),
        }
    )

    return df


@pytest.fixture
def sample_temporal_data():
    """
    Create sample temporal dataset for time-windowed encoding tests.

    Returns
    -------
    tuple
        (X, y) with temporal features
    """
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "feature": np.random.randn(n_samples),
        }
    )

    # Create target with temporal pattern
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.8, 0.2]), name="target")

    return X, y


@pytest.fixture
def model_params():
    """
    Standard model parameters for testing.

    Returns
    -------
    dict
        Model hyperparameters
    """
    return {
        "classifier__learning_rate": 0.05,
        "classifier__max_depth": 3,
        "classifier__n_estimators": 10,  # Small for fast tests
        "classifier__random_state": 42,
        "cat_encoder__strategy": "target_encoder",
    }
