"""
Feature engineering: date parts, drop leakage

- Loads cleaned dataset from `data/processed`
- Applies feature engineering
- Saves feature engineered datasets in `data/processed` 
"""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed_data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- feature functions ----------------------
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["year"] = df["Datetime"].dt.year
    df["month"] = df["Datetime"].dt.month
    df["day"] = df["Datetime"].dt.day
    df['hour'] = df["Datetime"].dt.hour

    # Reorder the columnns
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "month", df.pop("month"))
    df.insert(3, "day", df.pop("day")) 
    df.insert(4, "hour", df.pop("hour"))
    return df

# Drop unuse column(s)
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Drop columns that are not required for model training or inference.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with unused columns removed.
    """
    drop_cols = ["Datetime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

# ----------------------pipeline -----------------------
# handles full pipeline:
# Loads cleaned datasets -> applies feature engineering -> saves engineered datasets
def run_feature_engineering(
        in_train_path: Path | str | None = None,
        in_eval_path: Path | str | None = None,
        in_holdout_path: Path | str | None = None,
        output_dir: Path | str = PROCESSED_DIR
):
    """
    Run feature engineering and write outputs to disk.
    Applies the same transformations to train, eval, holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "clean_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "clean_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "clean_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)

    print("Train date range:", train_df["Datetime"].min(), "to", train_df["Datetime"].max())
    print("Eval date range:", eval_df["Datetime"].min(), "to", eval_df["Datetime"].max())
    print("Holdout date range:", holdout_df["Datetime"].min(), "to", holdout_df["Datetime"].max())

    # Date features
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    holdout_df = add_date_features(holdout_df)

    # Drop leakage/unused column(s)
    train_df = drop_unused_columns(train_df)
    eval_df = drop_unused_columns(train_df)
    holdout_df = drop_unused_columns(holdout_df)

    # Save engineered datasets
    out_train_path = output_dir / "feature_engineered_train.csv"
    out_eval_path = output_dir / "feature_engineered_eval.csv"
    out_holdout_path = output_dir / "feature_engineered_holdout.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print(" Feature engineering complete.")
    print(" Train shape:", train_df.shape)
    print(" Eval shape:", eval_df.shape)
    print(" Holdout shape:", holdout_df.shape)

    return train_df, eval_df, holdout_df

if __name__ == "__main__":
    run_feature_engineering()

