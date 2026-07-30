"""
============================================================
Project 5 - Ring Resonator Design Platform
Neural Network Evaluation
============================================================

Loads:
    - Trained neural network
    - Input scaler
    - Dataset

Evaluates:
    - MSE
    - RMSE
    - MAE
    - R² Score

Generates:
    - Metrics report
    - Metrics CSV
    - Prediction plots
============================================================
"""

import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

import config
from network import RingResonatorNetwork


# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------
# Load validation data
# ----------------------------------------------------------

def load_validation_data():

    print("Loading dataset...")

    dataframe = pd.read_csv(config.DATASET_PATH)

    X = dataframe.iloc[:, :4].values
    Y = dataframe.iloc[:, 4:].values

    _, X_validation, _, Y_validation = train_test_split(
        X,
        Y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )

    scaler = joblib.load(config.SCALER_PATH)

    X_validation = scaler.transform(X_validation)

    X_validation = torch.tensor(
        X_validation,
        dtype=torch.float32,
        device=DEVICE,
    )

    return X_validation, Y_validation


# ----------------------------------------------------------
# Load trained model
# ----------------------------------------------------------

def load_model():

    print("Loading trained model...")

    model = RingResonatorNetwork().to(DEVICE)

    checkpoint = torch.load(
        config.MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


# ----------------------------------------------------------
# Evaluate
# ----------------------------------------------------------

def evaluate_model(model, X_validation, Y_validation):

    print("Running inference...")

    with torch.no_grad():

        predictions = model(X_validation)

    predictions = predictions.cpu().numpy()

    output_scaler = joblib.load(
    config.OUTPUT_SCALER_PATH
    )

    predictions = output_scaler.inverse_transform(
    predictions
    )

    Y_validation = output_scaler.inverse_transform(
    Y_validation

    )

    mse = mean_squared_error(
        Y_validation,
        predictions,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        Y_validation,
        predictions,
    )

    r2 = r2_score(
    Y_validation,
    predictions,
    multioutput="variance_weighted"
)


    return predictions, mse, rmse, mae, r2, 


# ----------------------------------------------------------
# Save metrics
# ----------------------------------------------------------

def save_metrics(mse, rmse, mae, r2):

    results_directory = Path(config.RESULTS_DIRECTORY)

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = pd.DataFrame({

        "Metric": [

            "MSE",
            "RMSE",
            "MAE",
            "R2",
            

        ],

        "Value": [

            mse,
            rmse,
            mae,
            r2,

        ],

    })

    metrics.to_csv(
        results_directory / "evaluation_metrics.csv",
        index=False,
    )

    with open(
        results_directory / "evaluation_metrics.txt",
        "w",
    ) as file:

        file.write(
            "===============================\n"
        )

        file.write(
            "Neural Network Evaluation\n"
        )

        file.write(
            "===============================\n\n"
        )

        file.write(f"MSE  : {mse:.6f}\n")
        file.write(f"RMSE : {rmse:.6f}\n")
        file.write(f"MAE  : {mae:.6f}\n")
        file.write(f"R²   : {r2:.6f}\n")


# ----------------------------------------------------------
# Plot predictions
# ----------------------------------------------------------

def plot_predictions(
    predictions,
    ground_truth,
):

    results_directory = Path(config.RESULTS_DIRECTORY)

    random.seed(config.RANDOM_STATE)

    indices = random.sample(
        range(len(ground_truth)),
        5,
    )

    for count, index in enumerate(indices, start=1):

        plt.figure(figsize=(10, 5))

        plt.plot(
            ground_truth[index],
            label="Analytical",
            linewidth=2,
        )

        plt.plot(
            predictions[index],
            "--",
            label="Neural Network",
            linewidth=2,
        )

        plt.xlabel("Sample")

        plt.ylabel("Transmission")

        plt.title(
            f"Prediction Sample {count}"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            results_directory
            / f"prediction_sample_{count}.png",
            dpi=300,
        )

        plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("Neural Network Evaluation")
    print("=" * 60)

    X_validation, Y_validation = load_validation_data()

    model = load_model()

    (
        predictions,
        mse,
        rmse,
        mae,
        r2,
    ) = evaluate_model(
        model,
        X_validation,
        Y_validation,
    )

    save_metrics(
        mse,
        rmse,
        mae,
        r2,
    )

    plot_predictions(
        predictions,
        Y_validation,
    )

    print()

    print("Evaluation Complete")

    print()

    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R²   : {r2:.6f}")

    print()

    print(
        "Metrics saved to:",
        Path(config.RESULTS_DIRECTORY)
        / "evaluation_metrics.csv",
    )

    print(
        "Prediction plots saved to:",
        config.RESULTS_DIRECTORY,
    )


# ----------------------------------------------------------

if __name__ == "__main__":

    main()