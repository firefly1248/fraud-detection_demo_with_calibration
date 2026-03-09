RANDOM_SEED = 0
VALIDATION_N_SPLITS = 5

MODEL_FIXED_PARAMS = {
    "random_state": RANDOM_SEED,
    "deterministic": True,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "gain",
    "objective": "binary",
    "force_col_wise": True,
}

# Best params from Optuna (fraud_detection_optimization.db, trial #100, AUC-PR = 0.7781)
MODEL_DEFAULT_VARIABLE_PARAMS = {
    "cat_encoder__strategy": "target_encoder",
    "classifier__boosting_type": "gbdt",
    "classifier__learning_rate": 0.765,
    "classifier__max_depth": 8,
    "classifier__n_estimators": 163,
    "classifier__num_leaves": 100,
    "classifier__colsample_bytree": 0.634,
    "classifier__reg_alpha": 0.00154,
    "classifier__reg_lambda": 36.42,
    "classifier__min_split_gain": 6.4e-8,
}
