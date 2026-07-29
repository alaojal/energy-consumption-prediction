"""
Inference pipeline for energy consumption prediction.

This script:

- Loads raw input data.
- Applies preprocessing and feature engineering.
- Aligns features with the training schema.
- Loads a trained GradientBoostingRegressor model.
- Generates predictions.
- Saves the prediction results.
"""

# Raw data → Preprocessing → Feature Engineering → Align Features → Load Model → Predict → Save Predictions

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load 

# import preprocessing + feature engineering helpers
from src.data.preprocess import remove_outliers
from src.features.feature_engineering import add_date_features, drop_unused_columns

# ---------------------------------
# Default paths
# ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "GradientBoostingRegressor_best_model.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed_data" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

# Run inference from the command line.
if __name__ == "__main__":
    print(f"Inference using project root: {PROJECT_ROOT}")

# Load training features columns (strict schema from training dataset)
if not TRAIN_FE_PATH.exists():
    raise FileNotFoundError(
        f"Training feature file not found: {TRAIN_FE_PATH}"
    )

_train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "AEP_MW"] # exclude AEP_MW column




# ---------------------------
def predict (
        input_df: pd.DataFrame,
        model_path: Path | str = DEFAULT_MODEL
) -> pd.DataFrame:
    
    """
    Generate predictions for raw input data.

    Steps:
    1. Remove outliers.
    2. Apply feature engineering.
    3. Align features with the training schema.
    4. Load the trained model.
    5. Generate predictions.

    Args:
        input_df: Raw input DataFrame.
        model_path: Path to the trained model.

    Returns:
        pd.DataFrame containing the original input data
        with predicted values.    
    """
          
    # Step 1: preprocess
    df = remove_outliers(input_df)

    # Keep actual target values before removing target column
    y_true = None
    target = "AEP_MW"
    if target in df.columns:
        y_true = df[target].copy()

    # Step 2: Feature Engineering
    if "Datetime" in df.columns:
        df = add_date_features(df)
    
    # Drop unused columns
    df = drop_unused_columns(df)

    # Remove target before prediction
    if target in df.columns:
        df = df.drop(columns=[target])

    # Step 3: Align columns with training schema
    df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
                
    # Step 4: Load model and predict
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )
    model = load(model_path)

    # Step 5: Predict
    preds = model.predict(df)

    # Step 6: Build output  
    output = df.copy()

    if y_true is not None:
        output["actual_AEP_MW"] = y_true.values

    output["predicted_AEP_MW"] = preds

    return output
    

# ----------------------------
# CLI entrypoint
# ----------------------------
# Allows running inference directly from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new energy data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save prediction CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df, 
        model_path=args.model 
    )

    preds_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
