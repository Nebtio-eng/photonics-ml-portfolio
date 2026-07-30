"""
train_model.py

Train a Random Forest surrogate model for the
ring resonator dataset.
"""

import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.machine_learning.data_loader import (
    load_dataset,
    split_features_targets
)


def train_surrogate_model():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataframe = load_dataset()

    X, y = split_features_targets(dataframe)

    print(f"Number of samples : {len(dataframe)}")
    print(f"Input features    : {X.shape[1]}")
    print(f"Output targets    : {y.shape[1]}")

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = RandomForestRegressor(

        n_estimators=300,

        random_state=42,

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train

    )

    os.makedirs(

        "models",

        exist_ok=True

    )

    model_path = "models/ring_surrogate.joblib"

    joblib.dump(

        model,

        model_path

    )

    print("\nTraining complete.")

    print(f"Model saved to: {model_path}")

    return (

        model,

        X_test,

        y_test

    )


if __name__ == "__main__":

    train_surrogate_model()