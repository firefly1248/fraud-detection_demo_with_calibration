# Claude Instructions for Binary Classification Project

## Project Overview

Binary classification framework for fraud detection (IEEE-CIS, 590K transactions).

**Core Model**: `CalibratedBinaryClassifier`
**Calibration**: Isotonic, Venn-ABERS IVAP/CVAP (conformal prediction), Sigmoid
**Temporal Validation**: `TemporalGroupSplitter` with time-based splits

---

## Project Structure

```
calibrated_clf/
├── model.py               # CalibratedBinaryClassifier — core model + feature engineering
├── calibration.py         # MultiCalibrationWrapper, Venn-ABERS IVAP/CVAP, isotonic, sigmoid
├── data_loader.py         # load_fraud_data(), create_time_groups()
├── validators.py          # TemporalGroupSplitter (DO NOT MODIFY)
├── train_model.py         # Training pipeline, auto-detects TransactionDT for temporal CV
├── model_optimisation.py  # Optuna HP tuning — uses TemporalGroupSplitter when groups provided
├── feature_selection.py   # Recursive feature elimination
├── data_transformers.py   # FraudFeatureEngineer, TimeWindowedTargetEncoder
├── plot_functions.py      # Visualization utilities
└── config.py              # Fixed LightGBM params + best Optuna hyperparams (DO NOT MODIFY)

ieee-fraud-detection/      # Kaggle dataset — train/test_transaction.csv, train/test_identity.csv
build_and_evaluate_model.ipynb  # Fraud detection evaluation + calibration comparison
```

---

## Code Style

- **Type hints** on all function signatures
- **NumPy/Google docstrings** with Parameters, Returns, Examples sections
- **Sklearn conventions**: fitted attributes end in `_` (`model_`, `features_`, `is_fitted_`), inherit `BaseEstimator`/`ClassifierMixin`, return `self` from `fit()`
- **black** formatting enforced in CI — run `black calibrated_clf/` before committing

---

## Key Design Decisions

### Feature Engineering
- `prepare_and_extract_features()` auto-detects dataset from column names
- Applied automatically in `fit()` and `predict()` — no manual step needed
- To add features: edit `CalibratedBinaryClassifier.prepare_and_extract_features()` in `model.py`

### Pipeline Step Naming (`model.py`)
`CalibratedBinaryClassifier` wraps a sklearn `Pipeline` with two named steps: `cat_encoder` and `classifier`. All `variable_params` keys follow the `step_name__param` convention — e.g. `classifier__learning_rate`, `cat_encoder__strategy`. Breaking these names breaks `config.py`, Optuna, and all saved hyperparameter configs.

### Calibration: `MultiCalibrationWrapper`

**Methods:**
1. `isotonic` — fast, good default
2. `venn_abers` + `venn_abers_mode='inductive'` (IVAP) — 2 fits, reserves `cal_size` fraction; conformal intervals [p0, p1]
3. `venn_abers` + `venn_abers_mode='cross'` (CVAP) — `cv_folds+1` fits, uses 100% training data via OOF; better for small datasets
4. `sigmoid` — Platt scaling
5. `None` — no calibration

**Two-stage API:**
- `fit(X, y)` — splits internally, refits base model, fits calibrator
- `calibrate(X_cal, y_cal)` — calibrator only; base model must be pre-fitted (guarded by `check_is_fitted`); use when split is done externally (e.g. `compare_calibration_methods`)
- CVAP is only available via `fit()`, not `calibrate()`
- `calibrate()` prefers `base_estimator.classes_` over `np.unique(y_cal)` to handle imbalanced subsets

**⚠️ Important**: After temporal CV tuning on IEEE Fraud, LightGBM (binary cross-entropy) is already well-calibrated (ECE≈0.002). Post-hoc calibration on a temporally mismatched cal set tends to hurt AUC-PR. Always evaluate calibration on true out-of-time test set.

### Temporal Validation
- `TemporalGroupSplitter` requires `groups` (time bins from `create_time_groups(df, n_bins=50)`)
- Always set `gap_unique_groups > 0` to prevent leakage between train and val windows
- Optuna (`model_optimisation.py`) uses `TemporalGroupSplitter` automatically when `groups` is passed; falls back to `StratifiedKFold` otherwise
- **Old Optuna DB**: `load_if_exists=True` with stale DB imports old (StratifiedKFold) trials — delete DB before switching CV strategy

### TimeWindowedTargetEncoder
- Prevents both data leakage (only uses past) and concept drift (limits to recent window)
- Row-by-row processing: ~2-5 min for 590K rows at 30-day window
- `time_column` must remain in `X` throughout the pipeline

---

## Files NOT to Modify

- `validators.py` — `TemporalGroupSplitter` is correct and optimal
- `config.py` — fixed LightGBM params + best Optuna hyperparameters

**Key files for active development:**
- `model.py` — feature engineering, core model
- `calibration.py` — calibration methods and wrappers
- `train_model.py` — training pipeline

---

## Future Enhancements

- [ ] Unit tests
- [ ] Logging throughout pipeline
- [ ] Model monitoring / drift detection
- [ ] Temporal splits for CVAP (currently stratified k-fold only)
- [ ] FastAPI serving endpoint

---

**Last Updated**: 2026-03-11 | **Claude**: Sonnet 4.6+
