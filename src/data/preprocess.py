"""
Preprocessing script for the energy consumption forecasting project.

This script:

- Loads the training, evaluation and holdout datasets from `data/raw`.
- Removes extreme outliers from the target variable (`AEP_MW`).
- Saves cleaned datasets to `data/processed`.
"""
import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw_data")
PROCESSED_DIR = Path("data/processed_data")
OUTLIER_THRESHOLD = 22450
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Remove outlier
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove observations where AEP_MW exceeds the predefined threshold."""
    if "AEP_MW" not in df.columns:
        return df
    before = len(df)
    df = df[df["AEP_MW"] <= OUTLIER_THRESHOLD].copy()
    after = len(df)
    print(f"Removed {before - after} rows with AEP_MW > {OUTLIER_THRESHOLD}")
    return df

def preprocess_split(
        split: str,
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path | str = PROCESSED_DIR
) -> pd.DataFrame:
    """
    Load a dataset split, apply preprocessing, and save the cleaned dataset.

    Args:
        split: Dataset split name (e.g., "train", "eval", "holdout").
        raw_dir: Directory containing the raw CSV files.
        processed_dir: Directory where cleaned datasets will be saved.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
       
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    
    df = pd.read_csv(path)

    df = remove_outliers(df)

    out_path = processed_dir / f"clean_{split}.csv"
    df.to_csv(out_path, index=False)
    print(
        f"Preprocessed '{split}' dataset saved to {out_path} "
        f"with shape {df.shape}."
    )
        
    return df

def run_preprocess(
        splits: tuple[str, ...] = ("train", "eval", "holdout"),
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path | str = PROCESSED_DIR
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir)

if __name__ == "__main__":
    run_preprocess()







