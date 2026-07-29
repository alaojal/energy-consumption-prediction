import pandas as pd
import os


#def load_data(file_path: str) -> pd.DataFrame:
#    """
#    load CSV data into a pandas DataFrame.
#    Args:
#        file_path (str): Path to the CSV file
#    Return:
#        pd.DataFrame: Loaded dataset.
#   """
#    if not os.path.exists(file_path):
#        raise FileNotFoundError(f"File Not Found: {file_path}")
    
#    return pd.read_csv(file_path)


from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_data"
OUTPUT_DIR = RAW_DIR


def load_and_split_data(
        raw_path: Path | str = RAW_DIR / "AmericanElectricPower_hourly.csv",
        output_dir: Path | str = OUTPUT_DIR
):
    """    
    Load the raw dataset, split it into train, evaluation, and holdout
    datasets by date, and save the resulting CSV files.

    Args:
        raw_path: Path to the raw dataset.
        output_dir: Directory where the split datasets will be saved.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Training, evaluation, and holdout DataFrames.
      
    """

    raw_path = Path(raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"File not found: {raw_path.resolve()}")
    df = pd.read_csv(raw_path)

    # Convert Datetime column and sort chronologically
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    
    # Define split dates
    cutoff_date_eval = pd.Timestamp("2012-01-01") # validation start
    cutoff_date_holdout = pd.Timestamp("2016-01-01") # hold_out start

    # Splits the datasets
    train_df = df[df["Datetime"] < cutoff_date_eval]
    eval_df = df[(df["Datetime"] >= cutoff_date_eval) & (df["Datetime"] < cutoff_date_holdout)]
    holdout_df = df[df["Datetime"] >= cutoff_date_holdout]

    # Save the datasets
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    eval_df.to_csv(output_dir / "eval.csv", index=False)
    holdout_df.to_csv(output_dir / "holdout.csv", index=False)

    print(f"Data split completed and save to {output_dir.resolve()}")
    print(f" Train: {train_df.shape}")
    print(f" Eval: {eval_df.shape}")
    print(f"Holdout: {holdout_df.shape}")

    return train_df, eval_df, holdout_df

if __name__ == "__main__":
    load_and_split_data()