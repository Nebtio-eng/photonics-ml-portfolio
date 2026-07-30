"""
evaluate_model.py

Evaluate the trained Random Forest surrogate model.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from src.machine_learning.data_loader import (
    load_dataset,
    split_features_targets,
)


TARGET_NAMES = [
    "Average FSR (nm)",
    "Average FWHM (nm)",
    "Average Q",
    "Average ER (dB)",
    "Insertion Loss (dB)",
    "Average Finesse",
]


def evaluate_model():

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = joblib.load("models/ring_surrogate.joblib")

    print("Model loaded successfully.\n")

    dataframe = load_dataset()

    X, y = split_features_targets(dataframe)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("=" * 60)
    print("RUNNING PREDICTIONS")
    print("=" * 60)

    predictions = model.predict(X_test)

    os.makedirs("metrics", exist_ok=True)

    report_path = "metrics/evaluation_report.txt"

    with open(report_path, "w") as file:

        file.write("=" * 60 + "\n")
        file.write("MODEL EVALUATION REPORT\n")
        file.write("=" * 60 + "\n\n")

        for i, name in enumerate(TARGET_NAMES):

            r2 = r2_score(
                y_test.iloc[:, i],
                predictions[:, i]
            )

            mae = mean_absolute_error(
                y_test.iloc[:, i],
                predictions[:, i]
            )

            rmse = root_mean_squared_error(
                y_test.iloc[:, i],
                predictions[:, i]
            )

            print(f"{name}")
            print(f"R²   : {r2:.6f}")
            print(f"MAE  : {mae:.6f}")
            print(f"RMSE : {rmse:.6f}")
            print()

            file.write(f"{name}\n")
            file.write(f"R²   : {r2:.6f}\n")
            file.write(f"MAE  : {mae:.6f}\n")
            file.write(f"RMSE : {rmse:.6f}\n\n")

    print(f"Evaluation report saved to: {report_path}")

    return model, X_test, y_test, predictions


if __name__ == "__main__":

    evaluate_model()