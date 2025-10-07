import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle

# Load Excel files
trip_df = pd.read_excel("../data/trip_scenarios_clean.xlsx")
catalog_df = pd.read_excel("../data/ItemCatalog_clean.xlsx")

# Standardize column names
trip_df.columns = trip_df.columns.str.strip().str.lower()
catalog_df.columns = catalog_df.columns.str.strip().str.lower()

# Parse 'items' column into lists
trip_df["items"] = trip_df["items"].apply(
    lambda x: [int(i.strip()) for i in str(x).split(",") if i.strip().isdigit()]
)

# Define feature columns
categorical_cols = ["destination", "season", "weather", "activities"]

# Only use columns that exist
categorical_cols = [c for c in categorical_cols if c in trip_df.columns]

# intended numeric columns
intended_numeric_cols = [
    "duration_days", "avg_temp_high", "avg_temp_low",
    "rain_chance_percent", "humidity_percent", "uv_index", "altitude_m"
]

# Filter to existing numeric columns
numeric_cols = [col for col in intended_numeric_cols if col in trip_df.columns]
print("Using numeric columns:", numeric_cols)

# Encode categorical and numeric data
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Encode only if categorical_cols exist
if categorical_cols:
    encoded_cats = encoder.fit_transform(trip_df[categorical_cols])
    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)
else:
    encoded_cats = np.zeros((len(trip_df), 0))
    encoded_cat_cols = []

# Scale numeric features
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(trip_df[numeric_cols])

# Combine both feature sets
X = np.concatenate([scaled_nums, encoded_cats], axis=1)


# Multi-label encode items
mlb = MultiLabelBinarizer(classes=catalog_df["id"].tolist())
Y = mlb.fit_transform(trip_df["items"])


# Train/test split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)


# Define the MLP model
class TripItemMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
model = TripItemMLP(
    input_size=X_train.shape[1],
    hidden_size=128,
    output_size=Y_train.shape[1]
)


# Training setup
criterion = nn.BCEWithLogitsLoss()   # multi-label loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 25


# Train loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss/len(train_loader):.4f}")

#Evaluate model
model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(X_test))
    preds_binary = (preds > 0.5).float()
    accuracy = (preds_binary == Y_test).float().mean()
    print(f"\n Test Accuracy: {accuracy:.4f}")


# Save model + preprocessors-
torch.save(model.state_dict(), "trip_item_mlp.pth")

with open("preprocessors.pkl", "wb") as f:
    pickle.dump({
        "encoder": encoder,
        "scaler": scaler,
        "mlb": mlb,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }, f)

print("\nModel and preprocessors saved successfully!")

