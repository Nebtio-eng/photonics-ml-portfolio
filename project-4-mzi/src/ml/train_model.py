import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump

from src.ml.model import build_model


def train():
    df = pd.read_csv("results/mzi_dataset.csv")

    X = df[["L1", "L2", "neff", "wavelength"]]
    y = df["transmission"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    dump(model, "models/mzi_surrogate.joblib")

    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    print(f"MSE: {mse}")

    return model


if __name__ == "__main__":
    train()


