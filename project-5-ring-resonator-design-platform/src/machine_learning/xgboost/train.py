"""
train.py

Train the XGBoost surrogate model.
"""

import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# ==========================================================
# Paths
# ==========================================================

DATASET_PATH = "data/ring_resonator_dataset.csv"
MODEL_PATH = "models/xgboost_surrogate.joblib"

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATASET_PATH)

X = df.iloc[:, :4]
Y = df.iloc[:, 4:]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Build Model
# ==========================================================

base_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_model)

print("Training model...")

model.fit(X_train, Y_train)

print("Training complete.")

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")