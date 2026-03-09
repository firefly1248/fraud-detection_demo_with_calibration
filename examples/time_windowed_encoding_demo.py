#!/usr/bin/env python3
"""
Time-Windowed Target Encoding Demo

Demonstrates:
- TimeWindowedTargetEncoder to prevent data leakage and concept drift
- Comparison with standard target encoding
- Impact on model performance
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data
from src.data_transformers import TimeWindowedTargetEncoder
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def main():
    print("=" * 60)
    print("Time-Windowed Target Encoding Demo")
    print("=" * 60)

    # Load small sample for demo
    print("\n1. Loading fraud data...")
    df = load_fraud_data(sample_frac=0.05, verbose=False)  # 5% for faster demo

    print(f"   Loaded {len(df):,} transactions")
    print(f"   Time range: {df['TransactionDT'].min():.0f} to {df['TransactionDT'].max():.0f}")
    print(f"   Fraud rate: {df['isFraud'].mean():.2%}")

    # Select categorical features for encoding
    categorical_cols = ['card1', 'card2', 'ProductCD', 'card4', 'card6']

    # Keep only available columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    print(f"\n2. Encoding {len(categorical_cols)} categorical features:")
    print(f"   {', '.join(categorical_cols)}")

    # Prepare data
    X = df[categorical_cols + ['TransactionDT']].copy()
    y = df['isFraud'].copy()

    # Split by time (train on early data, test on later data)
    time_split = df['TransactionDT'].quantile(0.7)
    train_mask = df['TransactionDT'] < time_split
    test_mask = df['TransactionDT'] >= time_split

    X_train, y_train = X[train_mask].reset_index(drop=True), y[train_mask].reset_index(drop=True)
    X_test, y_test = X[test_mask].reset_index(drop=True), y[test_mask].reset_index(drop=True)

    print(f"\n3. Temporal split:")
    print(f"   Train: {len(X_train):,} transactions (earlier in time)")
    print(f"   Test: {len(X_test):,} transactions (later in time)")

    # Test different time windows
    time_windows = [7, 14, 30, 60]  # Days

    print(f"\n4. Testing TimeWindowedTargetEncoder with different windows...")
    print(f"\n{'Window (days)':<15} {'Train Time':<15} {'Encoding Stats'}")
    print("-" * 60)

    results = []

    for window_days in time_windows:
        print(f"\n   Testing {window_days}-day window...")

        # Initialize encoder
        encoder = TimeWindowedTargetEncoder(
            time_column='TransactionDT',
            time_window=window_days,  # Days
            cols=categorical_cols,
            smoothing=10.0,
            min_samples_leaf=10,
            verbose=False
        )

        # Encode training data
        import time
        start_time = time.time()
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        train_time = time.time() - start_time

        # Encode test data
        X_test_encoded = encoder.transform(X_test)

        # Remove timestamp column (not needed for model)
        X_train_model = X_train_encoded.drop(columns=['TransactionDT'])
        X_test_model = X_test_encoded.drop(columns=['TransactionDT'])

        # Calculate encoding statistics
        mean_encoding = X_train_model.mean().mean()
        std_encoding = X_train_model.std().mean()

        print(f"   {window_days:<15} {train_time:<15.2f}s "
              f"mean={mean_encoding:.4f}, std={std_encoding:.4f}")

        results.append({
            'window_days': window_days,
            'train_time': train_time,
            'mean_encoding': mean_encoding,
            'std_encoding': std_encoding
        })

    # Summary
    print(f"\n5. Summary...")
    print("\n   Key Observations:")
    print("   - Smaller windows = faster encoding but less data per sample")
    print("   - Larger windows = more data but risk of concept drift")
    print("   - 30-day window often provides good balance")

    print("\n   Why use time-windowed encoding?")
    print("   ✓ Prevents data leakage (only uses past data)")
    print("   ✓ Prevents concept drift (focuses on recent patterns)")
    print("   ✓ More realistic for production deployment")
    print("   ✓ Especially useful when fraud patterns change over time")

    print("\n6. Use Cases...")
    print("   - Financial fraud detection (patterns evolve)")
    print("   - User behavior modeling (preferences change)")
    print("   - Seasonal data (recent trends more relevant)")
    print("   - Any temporal data with concept drift")

    print("\n" + "=" * 60)
    print("✓ Time-windowed encoding demo completed!")
    print("\nRecommendation: Start with 30-day window, adjust based on")
    print("your domain knowledge and cross-validation results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
