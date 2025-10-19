# 480planit – Trip Packing Assistant (MLP)

This project uses a **Multi-Layer Perceptron (MLP)** model to suggest items travelers should bring on a trip.  
It takes trip details (destination, season, duration, weather, activities, etc.) and predicts the most likely items needed, grouped by category.

---

## 📂 Project Structure
<img width="471" height="328" alt="image" src="https://github.com/user-attachments/assets/786233e8-d99e-4964-9df2-c24e79b9cd0c" />



## ⚙️ Setup:
1. Clone the repository:
git clone https://github.com/mouzhass/480planit.git

2. Create a virtual environment:            
<img width="289" height="72" alt="image" src="https://github.com/user-attachments/assets/d0834f61-2119-4fb8-8bb6-825b97ccfd39" />


3. Install dependencies:
pip install -r requirements.txt


## 🏋️ Training the Model
python src/train_trip_mlp.py

-Reads data/trip_scenarios.xlsx

-Trains a multilabel classifier (MLP)
