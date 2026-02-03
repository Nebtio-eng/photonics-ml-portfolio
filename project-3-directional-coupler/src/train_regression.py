import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# -----------------------------
# Load dataset (ideal physics)
# -----------------------------
df = pd.read_csv("data/raw/coupler_dataset.csv")

X = df[["kappa", "length_um"]]
y = df["split_ratio"]

# -----------------------------
# Train ML surrogate
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
model.fit(X, y)

joblib.dump(model, "results/models/coupler_surrogate.pkl")
print("ML surrogate trained and saved")
