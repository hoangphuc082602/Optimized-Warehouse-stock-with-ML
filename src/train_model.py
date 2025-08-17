from __future__ import annotations

from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os
import joblib

from preprocess import (
    load_raw_data,
    build_dataset,
    split_features_target,
    FEATURES
)

def train_models(random_state: int = 42) -> Tuple[Pipeline, Pipeline, dict]:
    #Merge data
    train_df, center_df, meal_df = load_raw_data()
    data = build_dataset(train_df, center_df, meal_df)

    x = data[FEATURES]
    y= data["num_orders"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model_x = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    model_rfr = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

    # Fit
    xgb_model = model_x.fit(x_train, y_train)
    rfr_model = model_rfr.fit(x_train, y_train)

    # MAE & R²
    xgb_mae = mean_absolute_error(y_val, xgb_model.predict(x_val))
    xgb_r2 = r2_score(y_val, xgb_model.predict(x_val))

    rfr_mae = mean_absolute_error(y_val, rfr_model.predict(x_val))
    rfr_r2 = r2_score(y_val, rfr_model.predict(x_val))

    print(f"[XGBRegressor]  MAE={xgb_mae:.4f} | R2={xgb_r2:.4f}")
    print(f"[RandomForestReg]  MAE={rfr_mae:.4f} | R2={rfr_r2:.4f}")

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(ROOT_DIR, "Models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(rfr_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

    print("Models saved to ../Models/")

    context = {
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    return xgb_model, rfr_model, context


if __name__ == "__main__":
    _ = train_models()