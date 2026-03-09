#!/usr/bin/env python3
"""
Basic Fraud Detection Example

Demonstrates:
- Loading IEEE Fraud Detection data
- Training CalibratedBinaryClassifier with isotonic calibration
- Evaluating model performance with standard metrics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, create_time_groups
from src.model import CalibratedBinaryClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np


def main():
    print("=" * 60)
    print("Fraud Detection - Basic Example")
    print("=" * 60)

    # Load data (10% sample for quick demo)
    print("\n1. Loading IEEE Fraud Detection data...")
    df = load_fraud_data(sample_frac=0.1, verbose=True)

    # Create time groups for temporal validation
    print("\n2. Creating time groups...")
    df['time_group'] = create_time_groups(df, n_bins=50, verbose=False)

    # Prepare features and target
    print("\n3. Preparing features and target...")
    X = df.drop(columns=['isFraud', 'TransactionID', 'time_group'])
    y = df['isFraud']

    print(f"   Features shape: {X.shape}")
    print(f"   Fraud rate: {y.mean():.2%}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize model with isotonic calibration
    print("\n4. Training model with isotonic calibration...")
    model = CalibratedBinaryClassifier(
        variable_params={
            'classifier__learning_rate': 0.05,
            'classifier__max_depth': 6,
            'classifier__n_estimators': 100,
            'classifier__random_state': 42,
            'cat_encoder__strategy': 'target_encoder'
        },
        calibration_method='isotonic'
    )

    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")

    # Evaluate on test set
    print("\n5. Evaluating model performance...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print(f"\n{'Metric':<20} {'Score':<10}")
    print("-" * 30)
    print(f"{'AUC-ROC':<20} {auc_roc:.4f}")
    print(f"{'AUC-PR':<20} {auc_pr:.4f}")
    print(f"{'Brier Score':<20} {brier:.4f}")

    # Analyze predictions
    print("\n6. Prediction Analysis...")
    print(f"   Mean predicted probability: {y_pred_proba.mean():.4f}")
    print(f"   Actual fraud rate: {y_test.mean():.4f}")
    print(f"   Calibration gap: {abs(y_pred_proba.mean() - y_test.mean()):.4f}")

    # Flag high-risk transactions
    threshold = 0.5
    high_risk = (y_pred_proba > threshold).sum()
    print(f"\n   Transactions flagged as high-risk (>{threshold:.1%}): {high_risk} / {len(y_test)}")

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
