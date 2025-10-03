import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
DATA_DIR = BASE_DIR / "data"

TRIP_PATH = DATA_DIR / "trip_scenarios.xlsx"
CATALOG_PATH = DATA_DIR / "ItemCatalog.xlsx"


# -------------------------
# Helpers
# -------------------------
def norm_item(s: str) -> str:
    return str(s).strip().lower()


def parse_item_list(cell):
    if pd.isna(cell):
        return []
    s = str(cell)
    for sep in ['|', ';', ',']:
        if sep in s:
            return [norm_item(p) for p in s.split(sep) if str(p).strip()]
    return [norm_item(s)] if s.strip() else []


def detect_items_target(trip_df, catalog_items):
    """
    Build target Y (multi-hot matrix of items).
    Always uses human-readable item names, ignores raw numeric IDs.
    """
    candidates = [c for c in trip_df.columns if any(k in str(c).lower()
                                                    for k in ["item", "packed", "bring", "included"])]
    items_col = None
    for c in candidates:
        if trip_df[c].dtype == object:
            items_col = c
            break

    if items_col is not None:
        scenario_items = trip_df[items_col].apply(parse_item_list)

        # Start with catalog items (already names)
        all_items = set(catalog_items)

        # Add extra from scenarios, but skip if it's just digits
        for lst in scenario_items:
            for it in lst:
                if it and not it.isdigit():
                    all_items.add(it)

        y_labels = sorted(all_items)  # final label names
        y = np.zeros((len(trip_df), len(y_labels)), dtype=np.float32)
        idx = {it: i for i, it in enumerate(y_labels)}
        for r, lst in enumerate(scenario_items):
            for it in lst:
                if it in idx:
                    y[r, idx[it]] = 1.0
        return y, y_labels, items_col

    # fallback: many boolean columns
    bool_cols = [c for c in trip_df.columns if str(trip_df[c].dtype) in ["bool", "boolean"]]
    if bool_cols:
        y = trip_df[bool_cols].astype(float).values
        return y, bool_cols, None

    raise RuntimeError("No valid items column found in trip_scenarios.xlsx")


def build_preprocessor(trip_df, drop_cols):
    X = trip_df.drop(columns=drop_cols, errors="ignore").copy()
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("cat", "drop", []),
            ("num", StandardScaler(), num_cols) if num_cols else ("num", "drop", []),
        ],
        remainder="drop"
    )
    return pre, cat_cols, num_cols


# -------------------------
# MLP Model
# -------------------------
class MLPMultiLabel(nn.Module):
    def __init__(self, input_dim, hidden=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Main
# -------------------------
def main():
    # Load Excel files
    trip_df = pd.read_excel(TRIP_PATH, sheet_name=0)
    catalog_df = pd.read_excel(CATALOG_PATH, sheet_name=0)

    # catalog items
    name_cols = [c for c in catalog_df.columns if any(k in str(c).lower() for k in ["item", "name", "title"])]
    cat_name_col = name_cols[0] if name_cols else catalog_df.columns[0]
    catalog_items = [norm_item(x) for x in catalog_df[cat_name_col].astype(str).fillna("") if str(x).strip()]

    # build targets
    y, y_labels, items_col = detect_items_target(trip_df, catalog_items)

    # build features
    drop_cols = [items_col] if items_col else y_labels
    pre, cat_cols, num_cols = build_preprocessor(trip_df, drop_cols)
    X_processed = pre.fit_transform(trip_df.drop(columns=drop_cols, errors="ignore"))

    # feature names
    feature_names = []
    if cat_cols:
        ohe = pre.named_transformers_["cat"]
        if hasattr(ohe, "get_feature_names_out"):
            feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    if num_cols:
        feature_names.extend(num_cols)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = MLPMultiLabel(input_dim=input_dim, hidden=128, output_dim=output_dim)

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
    EPOCHS = 20
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in loader:
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {ep+1:02d}/{EPOCHS} | loss {total/len(X_train_t):.4f}")

    # eval
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "micro_f1": float(f1_score(y_test, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_test, preds, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_test, preds, average="micro", zero_division=0)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(input_dim),
        "n_labels": int(output_dim),
    }
    print("Final Metrics:", json.dumps(metrics, indent=2))

    # save artifacts
    outdir = BASE_DIR / "artifacts"
    outdir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), outdir / "model.pt")
    with open(outdir / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)
    with open(outdir / "labels.json", "w") as f:
        json.dump(y_labels, f, indent=2)   # <-- always human-readable names
    with open(outdir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print("Artifacts saved in", outdir)


if __name__ == "__main__":
    main()
