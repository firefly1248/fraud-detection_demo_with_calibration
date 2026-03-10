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

# Best params from Optuna (fraud_detection_optimization.db, temporal CV, AUC-PR = 0.5063)
MODEL_DEFAULT_VARIABLE_PARAMS = {
    "cat_encoder__strategy": "target_encoder",
    "classifier__boosting_type": "goss",
    "classifier__learning_rate": 0.07361804205513668,
    "classifier__max_depth": 8,
    "classifier__n_estimators": 166,
    "classifier__num_leaves": 111,
    "classifier__colsample_bytree": 0.6417147630495295,
    "classifier__reg_alpha": 5.143443569535221e-07,
    "classifier__reg_lambda": 3.838736308535599e-05,
    "classifier__min_split_gain": 1.0471657118126228e-08,
    "classifier__top_rate": 0.18410976659325012,
    "classifier__other_rate": 0.3808023999284922,
}
