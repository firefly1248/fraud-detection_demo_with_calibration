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
