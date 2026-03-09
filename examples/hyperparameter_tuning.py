#!/usr/bin/env python3
"""
Hyperparameter Tuning Example

Demonstrates:
- Automated hyperparameter optimization with Optuna
- Temporal cross-validation
- Model selection and saving
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_fraud_data, create_time_groups
from src.train_model import train_model
import joblib


def main():
    print("=" * 60)
    print("Hyperparameter Tuning with Optuna")
    print("=" * 60)

    # Load data
    print("\n1. Loading fraud detection data...")
    df = load_fraud_data(sample_frac=0.1, verbose=True)  # 10% sample for reasonable runtime

    # Create time groups
    print("\n2. Creating time groups for temporal validation...")
    df['time_group'] = create_time_groups(df, n_bins=50, verbose=False)

    print(f"   Created {df['time_group'].nunique()} time bins")

    # Run hyperparameter optimization
    print("\n3. Starting hyperparameter optimization...")
    print("   This will take several minutes...")
    print("   - Optimization trials: 20 (adjust n_trials for more/less)")
    print("   - CV splits: 3 (temporal cross-validation)")
    print("   - Metric: AUC-PR (better than AUC-ROC for imbalanced data)")

    # Train with HP optimization
    model = train_model(
        train_data=df,
        target_column='isFraud',
        with_hp_opt=True,
        n_trials=20,  # Increase for better results (but longer runtime)
        calibration_method='venn_abers',
        model_save_path='best_fraud_model.joblib'
    )

    print("\n4. Optimization completed!")
    print("   ✓ Best model saved to 'best_fraud_model.joblib'")

    # Show model details
    print("\n5. Model Details...")
    print(f"   Calibration method: {model.calibration_method}")
    print(f"   Fitted: {model.is_fitted_}")

    # Example: Load and use saved model
    print("\n6. Loading saved model...")
    loaded_model = joblib.load('best_fraud_model.joblib')
    print("   ✓ Model loaded successfully")

    # Quick test
    print("\n7. Testing loaded model...")
    X_sample = df.drop(columns=['isFraud', 'TransactionID', 'time_group']).head(5)
    predictions = loaded_model.predict_proba(X_sample)

    print("   Sample predictions (first 5 transactions):")
    for i, pred in enumerate(predictions[:, 1]):
        print(f"   Transaction {i+1}: {pred:.4f} fraud probability")

    print("\n" + "=" * 60)
    print("✓ Hyperparameter tuning example completed!")
    print("\nNext steps:")
    print("- Increase n_trials for better optimization (e.g., 100-200)")
    print("- Use full dataset (sample_frac=1.0) for production model")
    print("- Deploy 'best_fraud_model.joblib' to production")
    print("=" * 60)


if __name__ == "__main__":
    main()
