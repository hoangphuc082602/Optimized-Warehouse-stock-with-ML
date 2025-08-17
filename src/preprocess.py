from __future__ import annotations

import os
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

# PATHS 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data")

CENTER_INFO_CSV = os.path.join(DATA_DIR, "fulfilment_center_info.csv")
MEAL_INFO_CSV   = os.path.join(DATA_DIR, "meal_info.csv")
TRAIN_CSV       = os.path.join(DATA_DIR, "train.csv")

# ---- FEATURE SET ----
FEATURES: List[str] = [
    "id",  # cân nhắc loại bỏ nếu không có ý nghĩa dự báo
    "week",
    "center_id",
    "meal_id",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
]

def load_raw_data(
    train_csv: str = TRAIN_CSV,
    center_csv: str = CENTER_INFO_CSV,
    meal_csv: str = MEAL_INFO_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    center_info = pd.read_csv(center_csv)
    meal_info = pd.read_csv(meal_csv)
    train = pd.read_csv(train_csv)
    return train, center_info, meal_info

def build_dataset(
    train: pd.DataFrame, center_info: pd.DataFrame, meal_info: pd.DataFrame
) -> pd.DataFrame:
    data = (
        train.merge(center_info, on="center_id", how="inner")
             .merge(meal_info,   on="meal_id",   how="inner")
             .sort_values(by=["week"])
             .reset_index(drop=True)
    )
    return data

def split_features_target(
    data: pd.DataFrame,
    features: List[str] = FEATURES,
    target_col: str = "num_orders",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    # Kiểm tra cột bắt buộc
    missing = [c for c in features + [target_col] if c not in data.columns]
    if missing:
        raise KeyError(f"Các cột sau thiếu trong DataFrame: {missing}")

    X = data[features]
    y = data[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=random_state
    )

    # (tuỳ chọn) in kích thước để debug
    print("Shapes:",
          "x_train", x_train.shape,
          "x_val",   x_val.shape,
          "x_test",  x_test.shape,
          "y_train", y_train.shape,
          "y_val",   y_val.shape,
          "y_test",  y_test.shape)

    return x_train, x_val, x_test, y_train, y_val, y_test
