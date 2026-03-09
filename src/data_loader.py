"""
Data Loading and Preprocessing for IEEE Fraud Detection Dataset

This module handles:
- Loading and merging transaction + identity data
- Creating time groups for temporal validation
- Sampling for faster development iterations
"""

import typing as tp
import pandas as pd
import numpy as np
from pathlib import Path


def load_fraud_data(
    train_transaction_path: str = "ieee-fraud-detection/train_transaction.csv",
    train_identity_path: str = "ieee-fraud-detection/train_identity.csv",
    sample_frac: tp.Optional[float] = None,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and merge IEEE Fraud Detection transaction and identity data.

    Args:
        train_transaction_path: Path to transaction CSV file
        train_identity_path: Path to identity CSV file
        sample_frac: Optional fraction to sample (e.g., 0.1 for 10%).
                     Useful for faster development. None = use all data.
        random_state: Random seed for reproducible sampling
        verbose: Print loading progress and statistics

    Returns:
        Merged DataFrame with all features and target column 'isFraud'

    Example:
        >>> # Load 10% sample for development
        >>> df = load_fraud_data(sample_frac=0.1)
        >>>
        >>> # Load full dataset for training
        >>> df = load_fraud_data()
    """
    if verbose:
        print("=" * 80)
        print("Loading IEEE Fraud Detection Dataset")
        print("=" * 80)

    # Load transaction data
    if verbose:
        print(f"\n📂 Loading transactions from: {train_transaction_path}")

    trans_df = pd.read_csv(train_transaction_path)

    if verbose:
        print(f"   ✓ Loaded {len(trans_df):,} transactions with {trans_df.shape[1]} features")
        print(f"   ✓ Fraud rate: {trans_df['isFraud'].mean():.2%}")

    # Load identity data
    if verbose:
        print(f"\n📂 Loading identity data from: {train_identity_path}")

    identity_df = pd.read_csv(train_identity_path)

    if verbose:
        print(
            f"   ✓ Loaded {len(identity_df):,} identity records with {identity_df.shape[1]} features"
        )
        print(f"   ✓ Identity coverage: {len(identity_df)/len(trans_df)*100:.1f}% of transactions")

    # Merge on TransactionID (left join to keep all transactions)
    if verbose:
        print(f"\n🔗 Merging transaction and identity data...")

    df = trans_df.merge(identity_df, on="TransactionID", how="left")

    if verbose:
        print(f"   ✓ Merged shape: {df.shape}")
        print(
            f"   ✓ Missing values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / df.size * 100:.1f}%)"
        )

    # Sample if requested
    if sample_frac is not None:
        if verbose:
            print(f"\n📊 Sampling {sample_frac*100:.1f}% of data...")

        # Stratified sampling to preserve fraud rate
        from sklearn.model_selection import train_test_split

        _, df = train_test_split(
            df, test_size=sample_frac, random_state=random_state, stratify=df["isFraud"]
        )

        if verbose:
            print(f"   ✓ Sampled {len(df):,} transactions")
            print(f"   ✓ Fraud rate after sampling: {df['isFraud'].mean():.2%}")

    if verbose:
        print(f"\n✅ Data loading complete!")
        print(f"   Final shape: {df.shape}")
        print("=" * 80)

    return df


def create_time_groups(
    df: pd.DataFrame,
    time_column: str = "TransactionDT",
    n_bins: int = 50,
    method: str = "quantile",
    verbose: bool = True,
) -> pd.Series:
    """
    Create time-based groups for temporal validation using SimpleSplitter.

    Args:
        df: DataFrame with time column
        time_column: Name of timestamp column
        n_bins: Number of time bins to create
        method: Binning method
            - 'quantile': Equal-sized bins (same number of samples per bin)
            - 'uniform': Equal-width bins (same time duration per bin)
        verbose: Print binning statistics

    Returns:
        Series of integer time group labels (0 to n_bins-1)

    Example:
        >>> df['time_group'] = create_time_groups(df, n_bins=50)
        >>>
        >>> # Use with SimpleSplitter
        >>> splitter = SimpleSplitter(n_splits=5, val_unique_groups=5, gap_unique_groups=2)
        >>> for train_idx, val_idx in splitter.split(X, y, groups=df['time_group']):
        >>>     # Train on earlier periods, validate on later
    """
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")

    if verbose:
        print(f"\n⏰ Creating time groups from '{time_column}'...")
        print(f"   Time range: {df[time_column].min()} to {df[time_column].max()}")
        print(
            f"   Duration: {df[time_column].max() - df[time_column].min():,} seconds "
            + f"(~{(df[time_column].max() - df[time_column].min()) / 86400:.1f} days)"
        )

    if method == "quantile":
        # Equal-sized bins (same number of samples)
        time_groups = pd.qcut(
            df[time_column], q=n_bins, labels=False, duplicates="drop"  # Handle duplicate edges
        )
    elif method == "uniform":
        # Equal-width bins (same time duration)
        time_groups = pd.cut(df[time_column], bins=n_bins, labels=False, duplicates="drop")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'quantile' or 'uniform'")

    if verbose:
        actual_bins = time_groups.nunique()
        print(f"   ✓ Created {actual_bins} time groups (requested: {n_bins})")
        print(f"   ✓ Method: {method}")
        print(
            f"   ✓ Samples per group: min={time_groups.value_counts().min()}, "
            + f"mean={time_groups.value_counts().mean():.0f}, max={time_groups.value_counts().max()}"
        )

    return time_groups


def get_categorical_features(df: pd.DataFrame) -> tp.List[str]:
    """
    Identify categorical features in IEEE Fraud Detection dataset.

    Args:
        df: Merged DataFrame with transaction and identity data

    Returns:
        List of categorical feature names

    Note:
        This identifies explicitly categorical columns. High-cardinality
        ID-like features (card1, card2, card3) may benefit from target
        encoding rather than one-hot encoding.
    """
    categorical_features = []

    # Explicit categorical features
    explicit_cats = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
    categorical_features.extend([col for col in explicit_cats if col in df.columns])

    # M-features (boolean T/F)
    m_features = [f"M{i}" for i in range(1, 10)]
    categorical_features.extend([col for col in m_features if col in df.columns])

    # Identity categorical features
    identity_cats = ["DeviceType", "DeviceInfo"]
    categorical_features.extend([col for col in identity_cats if col in df.columns])

    # Card features (high cardinality - consider as categorical for target encoding)
    # Uncomment if you want to treat these as categorical
    # card_features = ['card1', 'card2', 'card3']
    # categorical_features.extend([col for col in card_features if col in df.columns])

    return categorical_features


def get_fraud_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get summary statistics about the fraud detection dataset.

    Args:
        df: Merged fraud detection DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    info = {
        "n_samples": len(df),
        "n_features": df.shape[1],
        "fraud_rate": df["isFraud"].mean(),
        "n_fraud": df["isFraud"].sum(),
        "n_legitimate": (df["isFraud"] == 0).sum(),
        "missing_rate": df.isnull().sum().sum() / df.size,
        "time_range_seconds": df["TransactionDT"].max() - df["TransactionDT"].min(),
        "time_range_days": (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400,
        "n_categorical": len(get_categorical_features(df)),
    }

    # Identity coverage
    if "DeviceType" in df.columns:
        info["identity_coverage"] = df["DeviceType"].notna().mean()
    else:
        info["identity_coverage"] = 0.0

    return info


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a formatted summary of the fraud detection dataset.

    Args:
        df: Merged fraud detection DataFrame
    """
    info = get_fraud_dataset_info(df)

    print("\n" + "=" * 80)
    print("FRAUD DETECTION DATASET SUMMARY")
    print("=" * 80)
    print(f"\n📊 Dataset Size:")
    print(f"   Samples:     {info['n_samples']:,}")
    print(f"   Features:    {info['n_features']}")
    print(f"   Categorical: {info['n_categorical']}")

    print(f"\n🎯 Target Distribution:")
    print(f"   Fraud:       {info['n_fraud']:,} ({info['fraud_rate']:.2%})")
    print(f"   Legitimate:  {info['n_legitimate']:,} ({1-info['fraud_rate']:.2%})")

    print(f"\n⏰ Time Range:")
    print(
        f"   Duration:    {info['time_range_days']:.1f} days ({info['time_range_seconds']:,} seconds)"
    )

    print(f"\n💾 Data Quality:")
    print(f"   Missing:     {info['missing_rate']:.1%}")
    print(f"   Identity:    {info['identity_coverage']:.1%} coverage")

    print("=" * 80)


# Example usage and testing
if __name__ == "__main__":
    # Test data loading with sample
    print("\n🧪 Testing data loader...")

    # Load 10% sample
    df = load_fraud_data(sample_frac=0.1, verbose=True)

    # Create time groups
    time_groups = create_time_groups(df, n_bins=50, verbose=True)
    df["time_group"] = time_groups

    # Print summary
    print_dataset_summary(df)

    # Show categorical features
    cat_features = get_categorical_features(df)
    print(f"\n📋 Categorical Features ({len(cat_features)}):")
    for feat in cat_features:
        print(f"   - {feat}")

    print("\n✅ Data loader test complete!")
