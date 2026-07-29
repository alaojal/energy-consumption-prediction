"""
Evaluate a trained Gradient Boosting Regression model.

This script:
- Loads a trained GradientBoostingRegressor model.
- Loads the feature-engineered evaluation dataset.
- Generates predictions.
- Computes evaluation metrics (RMSE, MAE, and R²).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from joblib import load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_EVAL = Path("data/processed_data/feature_engineered_eval.csv")
DEFAULT_MODEL = Path("models/GradientBoostingRegressor_model.pkl")


def _maybe_sample( df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def eval_model(
        model_path: Path | str = DEFAULT_MODEL,
        eval_path: Path | str = DEFAULT_EVAL,
        sample_frac: Optional[float] = None,
        random_state: int = 42
)-> dict[str, float]:
    
    """
    Evaluate a trained GradientBoostingRegressor.

    Returns
    -------
    dict[str, float]
    Evaluation metrics including RMSE, MAE, and R².
    """
    eval_df = pd.read_csv(eval_path)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "AEP_MW"
    if target not in eval_df.columns:
        raise ValueError(f"{target} not found in evaluation dataset.")

    X_eval, y_eval  = eval_df.drop(columns=[target]), eval_df[target]
    
    model = load(model_path)
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    y_pred = model.predict(X_eval)

    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    print("\nEvaluation Metrics")
    print("-" * 25)
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.4f}")
    return metrics

if __name__ == "__main__":
    eval_model()

