# Examples

This directory contains runnable example scripts demonstrating various use cases of the Calibrated Binary Classifier framework.

## Available Examples

### 1. `fraud_detection_basic.py`
Quick start example for IEEE Fraud Detection dataset:
- Load fraud data
- Basic model training with isotonic calibration
- Evaluate performance metrics

```bash
python examples/fraud_detection_basic.py
```

### 2. `fraud_detection_with_venn_abers.py`
Advanced example using Venn-ABERS calibration:
- Train model with uncertainty quantification
- Predict with confidence intervals
- Flag high-uncertainty predictions for manual review

```bash
python examples/fraud_detection_with_venn_abers.py
```

### 3. `time_windowed_encoding_demo.py`
Demonstrate time-windowed target encoding:
- Prevent both data leakage and concept drift
- Compare with standard CatBoost encoding
- Visualize encoding behavior over time

```bash
python examples/time_windowed_encoding_demo.py
```

### 4. `hyperparameter_tuning.py`
Hyperparameter optimization with Optuna:
- Automated HP search
- Temporal cross-validation
- Save best model

```bash
python examples/hyperparameter_tuning.py
```

## Requirements

All examples require the IEEE Fraud Detection dataset. Download from:
https://www.kaggle.com/c/ieee-fraud-detection

Place the data files in `ieee-fraud-detection/` directory:
```
ieee-fraud-detection/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

## Notes

- Examples use `sample_frac=0.1` by default for quick execution
- Increase `sample_frac` for better results (but longer runtime)
- GPU acceleration available for LightGBM if installed
