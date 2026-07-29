"""
Train a baseline Gradient Boosting Regression model.

This script:
- Loads feature-engineered train and evaluation datasets.
- Trains a GradientBoostingRegressor.
- Evaluate model performance.
- Saves the trained model.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_TRAIN = Path("data/processed_data/feature_engineered_train.csv")
DEFAULT_EVAL = Path("data/processed_data/feature_engineered_eval.csv")
DEFAULT_OUT = Path("models/GradientBoostingRegressor_model.pkl")

def _maybe_sample( df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def train_model(
        train_path: Path | str = DEFAULT_TRAIN,
        eval_path: Path | str = DEFAULT_EVAL,
        model_output: Path | str = DEFAULT_OUT,
        model_params: Optional[Dict] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42
)-> tuple[GradientBoostingRegressor, dict[str, float]]:
    """
    Train baseline GradientBoostingRegressor and save model.

    Returns
    ---------
    model: GradientBoostingRegressor
    metrics: dict[str, float]
    """
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "AEP_MW"
    if target not in train_df.columns:
        raise ValueError(f"{target} not found in training dataset.")

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval  = eval_df.drop(columns=[target]), eval_df[target]
    
    params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_depth": 3,
    "min_impurity_decrease": 0.0,
    "random_state": 42,
    "tol": 1e-4,
    "ccp_alpha": 0.0
    }
    if model_params:
        params.update(model_params)
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    print(f" Model trained. Saved to {out}")
    print("\nEvaluation Metrics")
    print("-" * 25)
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.4f}")

    return model, metrics

if __name__ == "__main__":
    train_model()

