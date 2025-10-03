# 480planit – Trip Packing Assistant (MLP)

This project uses a **Multi-Layer Perceptron (MLP)** model to suggest items travelers should bring on a trip.  
It takes trip details (destination, season, duration, weather, activities, etc.) and predicts the most likely items needed, grouped by category.

---

## 📂 Project Structure

480planit/
│── data/ # Input data
│ ├── trip_scenarios.xlsx # Training examples (trip details + items)
│ ├── ItemCatalog.xlsx # Master list of items with categories
│
│── src/ # Source code
│ ├── train_trip_mlp.py # Train the MLP model
│ ├── predict_example.py # Run predictions for a sample trip
│
│── artifacts/ # Saved training results
│ ├── model.pt
│ ├── preprocessor.pkl
│ ├── labels.json
│
│── .gitignore
│── README.md


## ⚙️ Setup:

1. Clone the repository:
git clone https://github.com/mouzhass/480planit.git

2. Create a virtual environment:            
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt
Example requirements.txt:

torch
numpy
pandas
scikit-learn
openpyxl

## 🏋️ Training the Model
python src/train_trip_mlp.py

-Reads data/trip_scenarios.xlsx
-Trains a multilabel classifier (MLP)
-Saves artifacts (model, preprocessor, labels) to artifacts/

## 🔮 Making Predictions

python src/predict_example.py

