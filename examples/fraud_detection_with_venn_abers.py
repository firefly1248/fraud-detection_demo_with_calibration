#!/usr/bin/env python3
"""
Fraud Detection with Venn-ABERS Example

Demonstrates:
- Venn-ABERS calibration with prediction intervals
- Uncertainty quantification for high-stakes decisions
- Flagging ambiguous predictions for manual review
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, create_time_groups
from src.model import CalibratedBinaryClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd


def main():
    print("=" * 60)
    print("Fraud Detection - Venn-ABERS Calibration Example")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df = load_fraud_data(sample_frac=0.1, verbose=True)
    df['time_group'] = create_time_groups(df, n_bins=50, verbose=False)

    X = df.drop(columns=['isFraud', 'TransactionID', 'time_group'])
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train with Venn-ABERS calibration
    print("\n2. Training model with Venn-ABERS calibration...")
    model = CalibratedBinaryClassifier(
        variable_params={
            'classifier__learning_rate': 0.05,
            'classifier__max_depth': 6,
            'classifier__n_estimators': 100,
            'classifier__random_state': 42,
            'cat_encoder__strategy': 'target_encoder'
        },
        calibration_method='venn_abers',
        calibration_params={'cal_size': 0.2}
    )

    model.fit(X_train, y_train)
    print("   ✓ Model trained with Venn-ABERS calibration")

    # Get predictions with intervals
    print("\n3. Computing prediction intervals...")
    intervals = model.predict_proba_with_intervals(X_test)

    # Extract interval components
    p_lower = intervals['p_lower']
    p_upper = intervals['p_upper']
    p_combined = intervals['p_combined']
    interval_width = intervals['interval_width']

    # Evaluate performance
    print("\n4. Model Performance...")
    auc_roc = roc_auc_score(y_test, p_combined)
    auc_pr = average_precision_score(y_test, p_combined)

    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   AUC-PR: {auc_pr:.4f}")

    # Analyze uncertainty
    print("\n5. Uncertainty Analysis...")
    print(f"   Mean interval width: {interval_width.mean():.4f}")
    print(f"   Median interval width: {np.median(interval_width):.4f}")
    print(f"   Max interval width: {interval_width.max():.4f}")

    # Flag high-uncertainty predictions for manual review
    uncertainty_threshold = 0.1
    uncertain_mask = interval_width > uncertainty_threshold
    n_uncertain = uncertain_mask.sum()

    print(f"\n6. High-Uncertainty Predictions (width > {uncertainty_threshold})...")
    print(f"   Transactions requiring manual review: {n_uncertain} / {len(X_test)} ({n_uncertain/len(X_test):.1%})")

    if n_uncertain > 0:
        # Analyze uncertain predictions
        uncertain_fraud_rate = y_test[uncertain_mask].mean()
        certain_fraud_rate = y_test[~uncertain_mask].mean()

        print(f"   Fraud rate in uncertain cases: {uncertain_fraud_rate:.2%}")
        print(f"   Fraud rate in certain cases: {certain_fraud_rate:.2%}")

    # Create decision framework
    print("\n7. Decision Framework...")
    print("   Based on prediction intervals:")

    # Low risk: upper bound < 0.3
    low_risk = p_upper < 0.3
    # High risk: lower bound > 0.7
    high_risk = p_lower > 0.7
    # Uncertain: wide interval
    uncertain = interval_width > uncertainty_threshold

    print(f"   - Auto-approve (low risk): {low_risk.sum()} transactions")
    print(f"   - Auto-block (high risk): {high_risk.sum()} transactions")
    print(f"   - Manual review (uncertain): {uncertain.sum()} transactions")
    print(f"   - Standard review (medium risk): {(~low_risk & ~high_risk & ~uncertain).sum()} transactions")

    # Show example predictions
    print("\n8. Example Predictions (first 5 test samples)...")
    print(f"\n{'Lower':<8} {'Upper':<8} {'Combined':<10} {'Width':<8} {'Actual':<8} {'Decision'}")
    print("-" * 60)

    for i in range(min(5, len(X_test))):
        decision = "High Risk" if high_risk[i] else ("Low Risk" if low_risk[i] else ("Uncertain" if uncertain[i] else "Medium"))
        actual = "Fraud" if y_test.iloc[i] == 1 else "Legit"

        print(f"{p_lower[i]:<8.3f} {p_upper[i]:<8.3f} {p_combined[i]:<10.3f} "
              f"{interval_width[i]:<8.3f} {actual:<8} {decision}")

    print("\n" + "=" * 60)
    print("✓ Venn-ABERS example completed!")
    print("\nKey Takeaway: Prediction intervals help identify ambiguous cases")
    print("that benefit from human expert review, improving decision quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()
