# Calibrated Binary Classifier

> Binary classification framework with **Venn-ABERS conformal prediction**, temporal validation, and automated feature engineering.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🌟 Key Features

- **Advanced Calibration Methods**
  - 🎯 **Venn-ABERS**: Conformal prediction with mathematical validity guarantees
  - 📊 Isotonic regression (sklearn standard)
  - 📈 Platt scaling (sigmoid)
  - 🔍 **Uncertainty quantification** via prediction intervals

- **Architecture**
  - ⚡ LightGBM with automated hyperparameter tuning (Optuna)
  - 🔧 Stateful feature engineering (no test leakage)
  - 🔬 Recursive Feature Elimination (performance-based, feature-engine)
  - ⏱️ Temporal validation for time-series data
  - 🕒 **Time-windowed target encoding** (prevents leakage + concept drift)
  - 📦 Scikit-learn compatible API


---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/firefly1248/fraud-detection_demo_with_calibration.git
cd fraud-detection_demo_with_calibration

# Create environment with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# Or with poetry
poetry install
poetry shell
```

### Basic Usage

```python
from calibrated_clf.data_loader import load_fraud_data, create_time_groups
from calibrated_clf.model import CalibratedBinaryClassifier

# Load IEEE Fraud Detection data
df = load_fraud_data(sample_frac=0.1)  # 10% sample for quick start
df['time_group'] = create_time_groups(df, n_bins=50)

# Prepare features
X = df.drop(columns=['isFraud', 'TransactionID', 'time_group'])
y = df['isFraud']

# Train with Venn-ABERS calibration
model = CalibratedBinaryClassifier(
    variable_params={
        'classifier__learning_rate': 0.05,
        'classifier__max_depth': 6,
        'classifier__n_estimators': 100,
        'cat_encoder__strategy': 'target_encoder'
    },
    calibration_method='venn_abers',
    calibration_params={'cal_size': 0.2}
)

# Fit model (feature engineering is automatic)
model.fit(X, y)

# Predict with uncertainty intervals
intervals = model.predict_proba_with_intervals(X_test)
print(f"Mean uncertainty: {intervals['interval_width'].mean():.4f}")

# Flag high-uncertainty predictions for manual review
uncertain = intervals['interval_width'] > 0.1
print(f"Uncertain predictions: {uncertain.sum()} / {len(X_test)}")
```

---

## 🎯 What Makes This Special?

### 1. Venn-ABERS Calibration ✨

Unlike standard calibration methods, **Venn-ABERS provides prediction intervals** with mathematical guarantees:

```python
# Standard calibration (point estimates)
model_isotonic = CalibratedBinaryClassifier(
    params, calibration_method='isotonic'
)
probas = model_isotonic.predict_proba(X_test)  # Just probabilities

# Venn-ABERS (prediction intervals)
model_venn = CalibratedBinaryClassifier(
    params, calibration_method='venn_abers'
)
intervals = model_venn.predict_proba_with_intervals(X_test)
# Returns: p_lower, p_upper, p_combined, interval_width
```

**When to use Venn-ABERS:**
- 🏥 High-stakes decisions (medical diagnosis, fraud detection, loan approval)
- 📊 Need uncertainty quantification beyond point estimates
- ⚖️ Distribution-free guarantees regardless of data characteristics
- 🚨 Alert systems where wide intervals trigger human review

### 2. Temporal Validation 📅

Built-in support for time-series cross-validation with **TemporalGroupSplitter**:

```python
from calibrated_clf.validators import TemporalGroupSplitter

splitter = TemporalGroupSplitter(
    n_splits=5,
    val_unique_groups=5,      # ~10% of data for validation
    gap_unique_groups=2,      # 2-bin gap prevents data leakage
    train_accounts_share=0
)

for train_idx, val_idx in splitter.split(X, y, groups=df['time_group']):
    # Train on past data, validate on future
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict_proba(X.iloc[val_idx])
```

**Why it matters:**
- ✅ Mimics production scenario (train on past, predict future)
- ✅ Prevents data leakage with temporal gap
- ✅ Realistic performance estimates

### 3. Automated Feature Engineering 🔧

Detects dataset type and applies domain-specific features automatically:

**Fraud Detection** (13 features):
- Transaction amount: log transform, decimal patterns
- Card aggregations: mean/std per card
- Time features: hour, day, weekday
- Email/address matching
- Missing value indicators

**No manual feature engineering required!**

---

## 📊 Performance Benchmarks

### IEEE Fraud Detection (3.5% fraud rate)

Calibration method comparison on held-out test set (temporal split, full 590K dataset):

| Method | AUC-PR | Brier Score | Log Loss | ECE |
|--------|--------|-------------|----------|-----|
| **Uncalibrated** | **0.5063** | **0.0229** | **0.0946** | **0.0024** |
| **Isotonic** | 0.4935 | 0.0231 | 0.0955 | 0.0056 |
| **Venn-ABERS** | 0.4957 | 0.0231 | 0.0950 | 0.0061 |
| **Sigmoid** | 0.5063 | 0.0239 | 0.1051 | 0.0146 |

> After temporal CV hyperparameter tuning, LightGBM is already well-calibrated (ECE=0.0024). Post-hoc calibration overfits to the calibration split and slightly hurts generalization on the out-of-time test set.

> Reproduce: run `build_and_evaluate_model.ipynb` on the full IEEE-CIS dataset.

### Calibration Methods — Metric Heatmap

<p align="center">
  <img src="docs/metrics_heatmap.png" alt="Metrics Heatmap" width="800"/>
</p>

*Left: raw metric values. Right: normalised (1 = best) — higher bar means better performance on every axis. Generated by `build_and_evaluate_model.ipynb`.*

### Calibration Comparison

<p align="center">
  <img src="docs/calibration_comparison.png" alt="Calibration Comparison" width="700"/>
</p>

*Example showing isotonic vs Venn-ABERS calibration curves and uncertainty distributions.*

> **Note**: Run `build_and_evaluate_model.ipynb` to regenerate both plots.

---

## 🏗️ Architecture

```
calibrated_clf/
├── model.py                    # CalibratedBinaryClassifier (core)
├── calibration.py              # Venn-ABERS + multi-calibration wrapper
├── data_loader.py              # IEEE Fraud data loading & time groups
├── validators.py               # TemporalGroupSplitter for temporal validation
├── train_model.py              # Training pipeline with HP optimization
├── model_optimisation.py       # Optuna hyperparameter tuning
├── feature_selection.py        # Recursive feature elimination (custom)
├── data_transformers.py        # FraudFeatureEngineer, TimeWindowedTargetEncoder
└── config.py                   # Fixed model parameters & random seed
```

**Key Design Principles:**
1. ✅ **Scikit-learn compatible** - Follows sklearn API conventions
2. ✅ **Fully documented** - NumPy/Google style docstrings everywhere
3. ✅ **Type-safe** - Complete type hints with type aliases
4. ✅ **Tested** - 45 unit tests, CI on Python 3.11 and 3.12
5. ✅ **Robust** - Error handling, validation, logging

---

## 📖 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete project guide for future development
- **Docstrings** - Every class and function fully documented with examples

### Example Documentation

```python
class CalibratedBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible binary classifier with advanced calibration.

    Parameters
    ----------
    variable_params : dict
        Hyperparameters for the model pipeline
    calibration_method : str, default='isotonic'
        Calibration method: 'isotonic', 'venn_abers', 'sigmoid', or 'none'
    calibration_params : dict, optional
        Additional parameters for calibration

    Examples
    --------
    >>> model = CalibratedBinaryClassifier(
    >>>     variable_params={'classifier__learning_rate': 0.05},
    >>>     calibration_method='venn_abers'
    >>> )
    >>> model.fit(X_train, y_train)
    >>> intervals = model.predict_proba_with_intervals(X_test)
    """
```

---

## 🔬 Advanced Features

### Hyperparameter Optimization

```python
from calibrated_clf.train_model import train_model

model = train_model(
    train_data=df,
    target_column='isFraud',
    with_hp_opt=True,
    n_trials=100,
    calibration_method='venn_abers'
)
```

### Model Interpretability

```python
# SHAP values for feature importance
shap_values = model.calculate_shap_values(X_test)
top_features = shap_values.abs().mean().sort_values(ascending=False).head(10)
print("Top contributing features:")
print(top_features)
```

### Time-Windowed Target Encoding

For temporal data with concept drift, use `TimeWindowedTargetEncoder` to prevent both data leakage and outdated patterns:

```python
from calibrated_clf.data_transformers import TimeWindowedTargetEncoder
from datetime import timedelta

# Encode categorical features using only recent past data
encoder = TimeWindowedTargetEncoder(
    time_column='TransactionDT',
    time_window=timedelta(days=30),  # Only use last 30 days
    cols=['card1', 'card2', 'ProductCD'],
    smoothing=10.0,      # Smoothing for rare categories
    min_samples_leaf=20  # Min samples required in window
)

X_encoded = encoder.fit_transform(X_train, y_train)

# Unlike CatBoost encoder (uses all past data), this focuses on recent patterns
# Prevents: ❌ Data leakage (future → past)
#          ❌ Concept drift (using outdated patterns)
```

**Why use time-windowed encoding?**
- 🕒 Fraud patterns change over time - old data may be misleading
- 📊 Balances preventing leakage with using relevant recent data
- 🎯 Especially useful for financial fraud, user behavior, and seasonal patterns
- ⚡ Performance: ~2-5 min for 590K transactions with 30-day window

---

## 📊 Dataset Information

### IEEE-CIS Fraud Detection

- **Samples**: 590,540 transactions
- **Features**: 394 transaction + 41 identity = 435 total
- **Target**: `isFraud` (3.5% positive class - highly imbalanced!)
- **Time Range**: 182 days (TransactionDT in seconds)
- **Missing Values**: 45% (normal for this dataset, handled automatically)

**Download**: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

Place files in `ieee-fraud-detection/` directory:
```
ieee-fraud-detection/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

---

## 🧪 Testing

### Run Feature Engineering Test
```bash
python -c "
from calibrated_clf.data_loader import load_fraud_data
from calibrated_clf.data_transformers import FraudFeatureEngineer

df = load_fraud_data(sample_frac=0.01)
X = df.drop(columns=['isFraud'])
engineer = FraudFeatureEngineer()
X_eng = engineer.fit_transform(X, df['isFraud'])
print(f'Original: {X.shape[1]}, Engineered: {X_eng.shape[1]}')
"
```

### Run Data Loader Test
```bash
python calibrated_clf/data_loader.py
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code style (black, type hints, docstrings)
4. Add tests for new features
5. Submit a pull request

---

## 📚 References

**Venn-ABERS & Conformal Prediction:**
- [Vovk et al. (2015) - Large-scale probabilistic predictors](https://arxiv.org/pdf/1511.00213.pdf)
- [ip200/venn-abers GitHub](https://github.com/ip200/venn-abers)
- [Awesome Conformal Prediction](https://github.com/valeman/awesome-conformal-prediction)

**Fraud Detection:**
- [IEEE-CIS Fraud Detection Challenge](https://www.kaggle.com/c/ieee-fraud-detection)

**Calibration:**
- [Niculescu-Mizil & Caruana (2005) - Predicting Good Probabilities](https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf)

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{calibrated_binary_classifier,
  author = {Ekhlakov, Ilia},
  title = {Calibrated Binary Classifier: ML with Venn-ABERS Conformal Prediction},
  year = {2026},
  url = {https://github.com/firefly1248/fraud-detection_demo_with_calibration}
}
```

---

## 🏆 Highlights

- ✨ **Cutting-edge**: Venn-ABERS conformal prediction (few implementations exist)
- 📚 **Well-documented**: 650+ lines of professional docstrings
- 🔒 **Type-safe**: Complete type hints throughout
- 🧪 **Validated**: Handles real-world fraud detection (590K transactions)
- 🎓 **Educational**: Clear examples and comprehensive guides

---

## 📧 Contact

**Author**: Ilia Ekhlakov

**Project Link**: https://github.com/firefly1248/fraud-detection_demo_with_calibration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ using Python, LightGBM, and Venn-ABERS</b>
</p>

<p align="center">
  <sub>⭐ Star this repo if you find it useful!</sub>
</p>
