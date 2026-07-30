"""
evaluate.py

Evaluate the trained XGBoost surrogate model.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# Paths
# ==========================================================

DATASET_PATH = "data/ring_resonator_dataset.csv"
MODEL_PATH = "models/xgboost_v2/xgboost_surrogate.joblib"


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
    random_state=42,
)


# ==========================================================
# Load Model
# ==========================================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.\n")


# ==========================================================
# Predict
# ==========================================================

print("Predicting test set...")

Y_pred = model.predict(X_test)

print("Prediction shape :", Y_pred.shape)
print("Ground truth shape:", Y_test.shape)

print("\nFirst device")

print("True:")
print(Y_test.iloc[0, :10].values)

print("\nPredicted:")
print(Y_pred[0, :10])

print("Prediction complete.\n")


# ==========================================================
# Metrics
# ==========================================================

mae = mean_absolute_error(Y_test, Y_pred)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

r2 = r2_score(
    Y_test,
    Y_pred,
    multioutput="uniform_average"
)

print("=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"MAE  : {mae:.8f}")
print(f"RMSE : {rmse:.8f}")
print(f"R²   : {r2:.8f}")




print()

print("R² for wavelength 0:")
print(r2_score(Y_test.iloc[:, 0], Y_pred[:, 0]))

print()

print("R² for wavelength 512:")
print(r2_score(Y_test.iloc[:, 512], Y_pred[:, 512]))

print()

print("R² for wavelength 1023:")
print(r2_score(Y_test.iloc[:, 1023], Y_pred[:, 1023]))

plt.figure(figsize=(10,4))

plt.plot(
    Y_test.iloc[0].values,
    label="True"
)

plt.plot(
    Y_pred[0],
    "--",
    label="Predicted"
)

plt.legend()

plt.xlabel("Transmission Sample")

plt.ylabel("Transmission")

plt.tight_layout()

plt.show()