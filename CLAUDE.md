# Claude Instructions for Binary Classification Project

## Project Overview

This is a production-ready **binary classification framework** supporting:
- **Fraud Detection** (IEEE-CIS Fraud Detection dataset)
- **Bid-Win Prediction** (RTB advertising auctions)
- **Advanced Calibration**: Isotonic, Venn-ABERS (conformal prediction), Sigmoid
- **Temporal Validation**: SimpleSplitter for time-series cross-validation
- **Feature Engineering**: Automated detection and engineering for both datasets

**Core Model**: `CalibratedBinaryClassifier` (formerly `BidWinModel` - alias maintained for backward compatibility)

---

## Project Structure

```
.
├── src/
│   ├── model.py                    # Core: CalibratedBinaryClassifier
│   ├── calibration.py              # Venn-ABERS + isotonic + sigmoid
│   ├── data_loader.py              # IEEE Fraud data loading & time groups
│   ├── validators.py               # SimpleSplitter for temporal validation
│   ├── train_model.py              # Training pipeline with HP optimization
│   ├── model_optimisation.py       # Optuna hyperparameter tuning
│   ├── feature_selection.py        # Recursive feature elimination
│   ├── data_transformers.py        # Categorical encoders, imputers
│   ├── custom_metrics.py           # AUC-PR and custom metrics
│   ├── plot_functions.py           # Visualization utilities
│   └── config.py                   # Fixed model parameters
├── ieee-fraud-detection/
│   ├── train_transaction.csv       # 590K transactions, 394 features
│   ├── train_identity.csv          # 144K identity records, 41 features
│   ├── test_transaction.csv
│   └── test_identity.csv
├── build_and_evaluate_model.ipynb  # Bid-win evaluation + calibration comparison
├── pyproject.toml                  # Poetry/uv dependencies
├── poetry.lock
├── FRAUD_DETECTION_MIGRATION_SPEC.md  # Complete migration specification
└── CLAUDE.md                       # This file

```

---

## Quick Start

### Environment Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
/Users/ilia.ekhlakov/.local/bin/uv pip install -r requirements.txt

# Or using poetry
poetry install
poetry shell
```

### Train on Fraud Detection

```python
from src.data_loader import load_fraud_data, create_time_groups
from src.model import CalibratedBinaryClassifier

# Load data
df = load_fraud_data(sample_frac=0.1)  # 10% sample for development
df['time_group'] = create_time_groups(df, n_bins=50)

# Separate features and target
X = df.drop(columns=['isFraud', 'TransactionID', 'time_group'])
y = df['isFraud']

# Apply feature engineering (automatic in fit, but can be done manually)
X_eng = CalibratedBinaryClassifier.prepare_and_extract_features(X)

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
model.fit(X_eng, y)

# Predict with uncertainty intervals
intervals = model.predict_proba_with_intervals(X_test)
print(f"Mean uncertainty: {intervals['interval_width'].mean():.4f}")
```

### Temporal Validation

```python
from src.validators import SimpleSplitter

splitter = SimpleSplitter(
    n_splits=5,
    val_unique_groups=5,      # ~10% of 50 bins
    gap_unique_groups=2,      # 2-bin gap prevents leakage
    train_accounts_share=0
)

for train_idx, val_idx in splitter.split(X, y, groups=df['time_group']):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    # Train and evaluate
```

---

## Code Style & Best Practices

### 1. **Type Hints**
Always use type hints for function signatures:
```python
def load_data(path: str, sample_frac: Optional[float] = None) -> pd.DataFrame:
    ...
```

### 2. **Docstrings**
Use NumPy/Google style with Parameters, Returns, Examples, Notes:
```python
def train_model(data: pd.DataFrame, target: str = "isFraud") -> CalibratedBinaryClassifier:
    """
    Train binary classification model with calibration.

    Parameters
    ----------
    data : pd.DataFrame
        Training data with features and target
    target : str, default='isFraud'
        Name of target column

    Returns
    -------
    CalibratedBinaryClassifier
        Fitted model with calibration

    Examples
    --------
    >>> model = train_model(df, target='isFraud')
    >>> predictions = model.predict_proba(X_test)
    """
```

### 3. **Sklearn Conventions**
- Fitted attributes use trailing underscore: `model_`, `features_`, `is_fitted_`
- Inherit from `BaseEstimator` and `ClassifierMixin`
- Implement `fit()`, `predict()`, `predict_proba()`
- Return `self` from `fit()` for chaining

### 4. **Error Handling**
Provide informative error messages:
```python
if len(unique_classes) != 2:
    raise ValueError(
        f"Target must be binary. Found {len(unique_classes)} classes: {unique_classes}"
    )
```

### 5. **Backward Compatibility**
When renaming classes, keep aliases:
```python
BidWinModel = CalibratedBinaryClassifier  # Backward compatibility
```

---

## Key Design Decisions

### Feature Engineering is Automatic
- `prepare_and_extract_features()` detects dataset type (fraud vs bid-win)
- Applied automatically in `fit()` and `predict()`
- Can be called manually if needed

### Calibration Methods
1. **Isotonic** (default): Fast, works well for large datasets
2. **Venn-ABERS**: Provides prediction intervals with mathematical guarantees
3. **Sigmoid**: Platt scaling (logistic regression)
4. **None**: No calibration

**When to use Venn-ABERS:**
- High-stakes decisions (fraud, medical, loans)
- Need uncertainty quantification
- Want to flag ambiguous predictions

### Temporal Validation
- Use `SimpleSplitter` with `time_group` column
- Set `gap_unique_groups > 0` to prevent data leakage
- Mimics production scenario (train on past, predict future)

### Time-Windowed Target Encoding
The `TimeWindowedTargetEncoder` in `data_transformers.py` prevents both **data leakage** and **concept drift**:
- Like CatBoost encoder, only uses past data (prevents leakage)
- Unlike CatBoost, limits history to recent time window (prevents drift)
- For each row at timestamp T, uses only data from [T - window, T)

**When to use:**
- Temporal data with concept drift (fraud patterns change over time)
- Want to focus on recent behavior patterns
- Balance between preventing leakage and using relevant historical data

```python
from src.data_transformers import TimeWindowedTargetEncoder

encoder = TimeWindowedTargetEncoder(
    time_column='TransactionDT',
    time_window=30,  # 30 days lookback
    cols=['card1', 'card2', 'ProductCD'],
    smoothing=10.0,
    min_samples_leaf=20
)
X_encoded = encoder.fit_transform(X_train, y_train)
```

---

## Common Tasks

### Add New Features
Edit `CalibratedBinaryClassifier.prepare_and_extract_features()` in `src/model.py`:
```python
if 'new_column' in X_.columns:
    X_["new_feature"] = X_["new_column"].apply(transformation)
```

### Change Calibration Method
```python
# Isotonic (fast)
model = CalibratedBinaryClassifier(params, calibration_method='isotonic')

# Venn-ABERS (with uncertainty)
model = CalibratedBinaryClassifier(
    params,
    calibration_method='venn_abers',
    calibration_params={'cal_size': 0.2}
)
```

### Run Hyperparameter Optimization
```python
from src.train_model import train_model

train_model(
    train_data=df,
    target_column='isFraud',
    with_hp_opt=True,
    n_trials=100,
    model_config_path='fraud_model_params.yaml',
    model_save_path='fraud_model.joblib'
)
```

### Load Saved Model
```python
import joblib
model = joblib.load('fraud_model.joblib')
predictions = model.predict_proba(X_test)
```

### Use Time-Windowed Target Encoding
For temporal data with concept drift, use `TimeWindowedTargetEncoder`:
```python
from src.data_transformers import TimeWindowedTargetEncoder
from sklearn.pipeline import Pipeline

# Create encoder (replaces standard CatBoost encoder)
time_encoder = TimeWindowedTargetEncoder(
    time_column='TransactionDT',
    time_window=timedelta(days=30),  # or int (30) for days, or float for seconds
    cols=['card1', 'card2', 'ProductCD', 'card4', 'card6'],
    smoothing=10.0,      # Higher = more smoothing for rare categories
    min_samples_leaf=20,  # Min samples in window to compute encoding
    verbose=True
)

# Apply to data (time_column must be in X)
X_encoded = time_encoder.fit_transform(X_train, y_train)

# Or use in pipeline (note: requires time_column in X throughout)
pipeline = Pipeline([
    ('time_encoder', time_encoder),
    ('imputer', MissingDataHandler(strategy='mean')),
    ('classifier', lgb.LGBMClassifier())
])
```

**Performance Note**: Row-by-row processing can be slow for large datasets. For 590K transactions:
- time_window=30 days: ~2-5 min per epoch
- time_window=7 days: ~1-2 min per epoch
- Consider using smaller time window or sampling for development

---

## Dataset Information

### IEEE Fraud Detection
- **Samples**: 590,540 transactions
- **Features**: 394 transaction + 41 identity = 435 total
- **Target**: `isFraud` (3.5% positive class)
- **Time Range**: 182 days (TransactionDT in seconds)
- **Missing Values**: 45% (normal, handled by LightGBM)
- **Identity Coverage**: 24.4% of transactions

**Key Features:**
- `TransactionAmt`: Transaction amount in USD
- `TransactionDT`: Timestamp (seconds from reference)
- `ProductCD`: Product category (W, C, H, R, S)
- `card1-6`: Card information
- `P_emaildomain`, `R_emaildomain`: Email domains
- `C1-C14`: Count features
- `D1-D15`: Timedelta features
- `M1-M9`: Match features (boolean T/F)
- `V1-V339`: Vesta engineered features
- `DeviceType`, `DeviceInfo`: Device information

### Bid-Win Prediction (Legacy)
- **Target**: Binary bid win/loss
- **Features**: price, flr, sellerClearPrice, dsp, hour, lang, etc.
- **Maintained for backward compatibility**

---

## Testing

### Run Feature Engineering Test
```bash
.venv/bin/python3 -c "
from src.data_loader import load_fraud_data
from src.model import CalibratedBinaryClassifier

df = load_fraud_data(sample_frac=0.01)
X = df.drop(columns=['isFraud'])
X_eng = CalibratedBinaryClassifier.prepare_and_extract_features(X)
print(f'Original: {X.shape[1]}, Engineered: {X_eng.shape[1]}')
"
```

### Run Data Loader Test
```bash
.venv/bin/python3 src/data_loader.py
```

---

## Dependencies

**Core:**
- Python 3.11+
- LightGBM
- scikit-learn 1.4.0
- pandas
- numpy

**Feature Engineering:**
- category-encoders
- feature-engine

**Calibration:**
- venn-abers (Venn-ABERS conformal prediction)

**Optimization:**
- optuna

**Interpretation:**
- shap

**Visualization:**
- matplotlib
- seaborn

---

## Performance Benchmarks

### Fraud Detection (3.5% fraud rate)
| Metric | Expected |
|--------|----------|
| AUC-ROC | 0.93-0.97 |
| AUC-PR | 0.55-0.75 |
| Brier Score | <0.025 |
| ECE | <0.01 |

**Note**: Calibration is CRITICAL for imbalanced data!

---

## Files NOT to Modify

These files work correctly and should not be changed without good reason:
- `validators.py` - SimpleSplitter already perfect for temporal validation
- `config.py` - Fixed LightGBM parameters
- `custom_metrics.py` - AUC-PR implementations

---

## Recent Changes

**2026-02-08:**
- ✅ Implemented Venn-ABERS calibration (`src/calibration.py`)
- ✅ Added IEEE Fraud Detection data loader (`src/data_loader.py`)
- ✅ Refactored `BidWinModel` → `CalibratedBinaryClassifier` with backward compatibility
- ✅ Added comprehensive docstrings (NumPy/Google style)
- ✅ Added type hints throughout core model
- ✅ Updated feature engineering for fraud detection (13 new features)
- ✅ Added calibration comparison to notebook (6 new cells)
- ✅ Implemented `TimeWindowedTargetEncoder` in `data_transformers.py` for temporal target encoding with sliding window to prevent both data leakage and concept drift

---

## Troubleshooting

### LightGBM libomp.dylib Error (macOS)
```bash
# Install OpenMP
brew install libomp

# Or use poetry/uv to reinstall
uv pip install --force-reinstall lightgbm
```

### Memory Issues with Large Dataset
```python
# Use sampling for development
df = load_fraud_data(sample_frac=0.1)  # 10% sample

# Or reduce n_bins for time groups
df['time_group'] = create_time_groups(df, n_bins=20)  # Instead of 50
```

### Venn-ABERS Import Error
```bash
uv pip install venn-abers
```

---

## Future Enhancements

**High Priority:**
- [ ] Add logging throughout the pipeline
- [ ] Create comprehensive unit tests
- [ ] Add model monitoring/drift detection
- [ ] Containerize with Docker

**Medium Priority:**
- [ ] Add support for multiclass classification
- [ ] Implement automated feature selection in pipeline
- [ ] Add model explainability dashboard
- [ ] Create FastAPI serving endpoint

**Low Priority:**
- [ ] Support for additional calibration methods (Beta calibration)
- [ ] Add support for categorical target encoding strategies
- [ ] Implement automated hyperparameter tuning in production

---

## Contact & Support

**Project Type**: Binary Classification Framework
**Primary Use Cases**: Fraud Detection, Bid-Win Prediction
**Calibration**: Isotonic, Venn-ABERS (conformal prediction), Sigmoid
**Temporal Validation**: SimpleSplitter with time-based splits

**Key Files for Modification:**
1. `src/model.py` - Core model and feature engineering
2. `src/data_loader.py` - Data loading and preprocessing
3. `src/train_model.py` - Training pipeline

**Key Files for Reference Only:**
1. `src/validators.py` - Temporal validation (already optimal)
2. `src/config.py` - Fixed parameters (rarely changed)
3. `FRAUD_DETECTION_MIGRATION_SPEC.md` - Complete migration documentation

---

**Last Updated**: 2026-02-08
**Claude Code Version**: Compatible with Claude Sonnet 4.5+
