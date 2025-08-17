import os, sys
import joblib
import pandas as pd
from pathlib import Path
from preprocess import FEATURES

#Got model PATH
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
MODEL_DIR = ROOT_DIR / "Models"

#LOADING RandomForestRegression model
model_path = MODEL_DIR / "random_forest_model.pkl"
model = joblib.load(model_path)
print(f"Model loaded from {model_path}")

#Loading data
test_df   = pd.read_csv(DATA_DIR / "test.csv")
center_df = pd.read_csv(DATA_DIR / "fulfilment_center_info.csv")
meal_df   = pd.read_csv(DATA_DIR / "meal_info.csv")

test_full = (
    test_df.merge(center_df, on="center_id", how="left")
           .merge(meal_df,   on="meal_id",   how="left")
)

x_test = test_full[FEATURES]

y_pred = model.predict(x_test)

# Export Submission file
out = pd.DataFrame({
    "id": test_df["id"] if "id" in test_df.columns else range(len(test_df)),
    "num_orders_pred": y_pred
})

out_path = DATA_DIR / "submission.csv"
out.to_csv(out_path, index=False)
print("Predictions saved to", out_path)
