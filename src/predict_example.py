# ===========================
# predict_example.py
# ===========================

import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from train_trip_mlp import TripItemMLP  # Import model class

# ---------------------------
# 1. Load model and preprocessors
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # goes from src -> project root
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Load preprocessors
with open(MODEL_DIR / "preprocessors.pkl", "rb") as f:
    data = pickle.load(f)

encoder = data["encoder"]
scaler = data["scaler"]
mlb = data["mlb"]
categorical_cols = data["categorical_cols"]
numeric_cols = data["numeric_cols"]

# Load item catalog (for readable output)
catalog_df = pd.read_excel(DATA_DIR / "ItemCatalog_clean.xlsx")
catalog_df.columns = catalog_df.columns.str.strip().str.lower()

# Initialize model
input_size = len(numeric_cols) + len(encoder.get_feature_names_out(categorical_cols))
output_size = len(mlb.classes_)
model = TripItemMLP(input_size, 128, output_size)
model.load_state_dict(torch.load(MODEL_DIR / "trained_trip_item_mlp.pth"))
model.eval()

print("Model and preprocessors loaded successfully.\n")


# ---------------------------
# 2. Define a new trip example
# ---------------------------
new_trip = {
    "destination": "Miami, FL",
    "duration_days": 7,
    "season": "Summer",
    "weather": "Sunny",
    "avg_temp_high": 90,
    "avg_temp_low": 75,
    "rain_chance_percent": 20,
    "humidity_percent": 70,
    "activities": "Beach, Swimming"
}

# ---------------------------
# 3. Convert new trip to DataFrame
# ---------------------------
new_df = pd.DataFrame([new_trip])

# Encode categorical + scale numeric (must match training pipeline)
print("Encoded categories:", encoder.get_feature_names_out(categorical_cols))
encoded_cats = encoder.transform(new_df[categorical_cols])
scaled_nums = scaler.transform(new_df[numeric_cols])
X_new = np.concatenate([scaled_nums, encoded_cats], axis=1)

X_tensor = torch.tensor(X_new, dtype=torch.float32)

# ---------------------------
# 4. Predict with model
# ---------------------------
with torch.no_grad():
    preds = torch.sigmoid(model(X_tensor))[0]

# Convert to numpy for sorting
preds = preds.numpy()
sorted_indices = np.argsort(preds)[::-1]  # Sort descending by confidence

# ---------------------------
# 5. Print Top 10 Predictions
# ---------------------------
print("Top 10 Recommended Items:\n")
for i in range(10):
    item_id = mlb.classes_[sorted_indices[i]]
    item_info = catalog_df.loc[catalog_df["id"] == item_id]
    item_name = item_info["name"].values[0] if not item_info.empty else f"Item {item_id}"
    item_cat = item_info["category"].values[0] if not item_info.empty else "Unknown"
    confidence = preds[sorted_indices[i]]
    print(f"{i+1:02d}. {item_name:45s} | {item_cat:20s} | confidence = {confidence:.2f}")
