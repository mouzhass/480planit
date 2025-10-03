import json
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from train_trip_mlp import MLPMultiLabel, BASE_DIR, TRIP_PATH, CATALOG_PATH

# -------------------------
# Paths
# -------------------------
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "model.pt"
PREPROCESSOR_PATH = ARTIFACTS / "preprocessor.pkl"
LABELS_PATH = ARTIFACTS / "labels.json"

# -------------------------
# Load artifacts
# -------------------------
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# Load model
input_dim = len(preprocessor.get_feature_names_out())
output_dim = len(labels)
model = MLPMultiLabel(input_dim=input_dim, hidden=128, output_dim=output_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print(f"Loaded model with {input_dim} features and {output_dim} labels.")

# -------------------------
# Load Item Catalog for categories
# -------------------------
catalog_df = pd.read_excel(CATALOG_PATH, sheet_name=0)

# Guess catalog columns
name_col = [c for c in catalog_df.columns if "item" in c.lower() or "name" in c.lower()][0]
cat_col = [c for c in catalog_df.columns if "category" in c.lower()][0]

catalog_map = dict(zip(
    catalog_df[name_col].astype(str).str.lower(),
    catalog_df[cat_col]
))

# -------------------------
# Build input template
# -------------------------
trip_df = pd.read_excel(TRIP_PATH, sheet_name=0)
drop_cols = [c for c in trip_df.columns if "item" in str(c).lower()]
required_cols = trip_df.drop(columns=drop_cols, errors="ignore").columns

example_trip = pd.DataFrame([{col: "" if trip_df[col].dtype == object else 0 for col in required_cols}])

# Ensure missing columns (like item_names, item_categories, baseline_items) exist
expected_cols = set(preprocessor.feature_names_in_)
missing_cols = expected_cols - set(example_trip.columns)
for col in missing_cols:
    example_trip[col] = ""

# -------------------------
# Fill in only what you care about
# -------------------------
example_trip.loc[0, "destination"] = "Miami"
example_trip.loc[0, "season"] = "summer"
example_trip.loc[0, "weather"] = "sunny"
example_trip.loc[0, "duration_days"] = 7
example_trip.loc[0, "avg_temp_high"] = 90
example_trip.loc[0, "avg_temp_low"] = 75
example_trip.loc[0, "rain_chance_percent"] = 20
example_trip.loc[0, "humidity_percent"] = 70
example_trip.loc[0, "uv_index"] = 10
example_trip.loc[0, "altitude_m"] = 5
example_trip.loc[0, "activities"] = "swimming|boating"

print("\nExample trip row:")
print(example_trip.to_string(index=False))

# -------------------------
# Predict
# -------------------------
X_new = preprocessor.transform(example_trip)
X_new_t = torch.tensor(X_new, dtype=torch.float32)

with torch.no_grad():
    logits = model(X_new_t)
probs = torch.sigmoid(logits).numpy()[0]

# -------------------------
# Group predictions by category
# -------------------------
results = []
for i, prob in enumerate(probs):
    item_name = labels[i]
    item_name_norm = str(item_name).lower()
    category = catalog_map.get(item_name_norm, "Uncategorized")
    results.append((category, item_name, prob))

# Sort by probability
results.sort(key=lambda x: x[2], reverse=True)

# Print top-N grouped
top_k = 10
grouped = {}
for category, item_name, prob in results[:top_k]:
    grouped.setdefault(category, []).append((item_name, prob))

print("\nTop predicted items (grouped by category):")
for cat, items in grouped.items():
    print(f"\n{cat}:")
    for item, prob in items:
        print(f"- {item}: {prob:.2f}")
