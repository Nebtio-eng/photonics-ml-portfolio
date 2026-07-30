import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from network import RingResonatorNetwork
import config

def set_seed(seed: int = 42):
    """
    Makes training reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

def create_directories():
    """
    Creates output folders if they do not exist.
    """

    os.makedirs(
        config.MODEL_DIRECTORY,
        exist_ok=True
    )

    os.makedirs(
        config.RESULTS_DIRECTORY,
        exist_ok=True
    )

def get_device():
    """
    Returns the computation device.
    """

    if torch.cuda.is_available():

        print("Using GPU")

        return torch.device("cuda")

    print("Using CPU")

    return torch.device("cpu")

def prepare_data():
    """
    Loads the dataset, normalizes the inputs,
    and creates PyTorch DataLoaders.
    """

    print("Loading dataset...")

    # --------------------------------------------------
    # Read CSV
    # --------------------------------------------------

    df = pd.read_csv(config.DATASET_PATH)

    # --------------------------------------------------
    # Split inputs and outputs
    # --------------------------------------------------

    X = df.iloc[:, :4].values

    Y = df.iloc[:, 4:].values
    

    

    # --------------------------------------------------
    # Train / Validation split
    # --------------------------------------------------

    X_train, X_validation, Y_train, Y_validation = train_test_split(
    X,
    Y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
)

    # --------------------------------------------------
    # Normalize inputs
    # --------------------------------------------------
    
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_train = input_scaler.fit_transform(X_train)
    X_validation = input_scaler.transform(X_validation)

    Y_train = output_scaler.fit_transform(Y_train)
    Y_validation = output_scaler.transform(Y_validation)


    # --------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    Y_train = torch.tensor(
        Y_train,
        dtype=torch.float32
    )

    X_validation = torch.tensor(
        X_validation,
        dtype=torch.float32
    )

    Y_validation = torch.tensor(
        Y_validation,
        dtype=torch.float32
    )

    # --------------------------------------------------
    # Create datasets
    # --------------------------------------------------

    train_dataset = TensorDataset(
    X_train,
    Y_train,
    
)

    validation_dataset = TensorDataset(
    X_validation,
    Y_validation,
    
)
    # --------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    print()

    print(f"Training Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {len(validation_dataset)}")

    print()

    return (
    train_loader,
    validation_loader,
    input_scaler,
    output_scaler,)

class PhysicsWeightedMSE(nn.Module):

    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, prediction, target, physical_target):

        weights = 1.0 + self.alpha * (1.0 - physical_target)

        loss = weights * (prediction - target) ** 2

        return loss.mean()

    
def build_model(device):
    """
    Creates the neural network,
    loss function,
    and optimizer.
    """

    print("Building neural network...")

    model = RingResonatorNetwork()

    model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=1e-4
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()

    print(model)

    print()

    print(f"Total Parameters      : {total_parameters:,}")
    print(f"Trainable Parameters  : {trainable_parameters:,}")

    print()

    return (
    model,
    criterion,
    optimizer,
    scheduler
)
    

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    """
    Trains the model for one epoch.

    Returns
    -------
    float
        Average training loss.
    """

    # ------------------------------------------
    # Put model into training mode
    # ------------------------------------------

    model.train()

    running_loss = 0.0

    # ------------------------------------------
    # Loop through every mini-batch
    # ------------------------------------------

    for inputs, targets in train_loader:

        # -----------------------------
        # Move tensors to device
        # -----------------------------

        inputs = inputs.to(device)

        targets = targets.to(device)

        # -----------------------------
        # Clear previous gradients
        # -----------------------------

        optimizer.zero_grad()

        # -----------------------------
        # Forward pass
        # -----------------------------

        predictions = model(inputs)

        # -----------------------------
        # Compute loss
        # -----------------------------

        loss = criterion(
        predictions,
        targets.to(device)
        )

        # -----------------------------
        # Backpropagation
        # -----------------------------

        loss.backward()

        # -----------------------------
        # Update weights
        # -----------------------------

        optimizer.step()

        # -----------------------------
        # Accumulate loss
        # -----------------------------

        running_loss += loss.item() * inputs.size(0)

    # ------------------------------------------
    # Average loss
    # ------------------------------------------

    average_loss = running_loss / len(train_loader.dataset)

    return average_loss

def validate(
    model,
    validation_loader,
    criterion,
    device
):
    """
    Evaluates the model on the validation dataset.

    Returns
    -------
    float
        Average validation loss.
    """

    # ------------------------------------------
    # Put model into evaluation mode
    # ------------------------------------------

    model.eval()

    running_loss = 0.0

    # ------------------------------------------
    # Disable gradient calculations
    # ------------------------------------------

    with torch.no_grad():

        # --------------------------------------
        # Loop over every validation batch
        # --------------------------------------

        for inputs, targets in validation_loader:

            inputs = inputs.to(device)

            targets = targets.to(device)

            # -----------------------------
            # Forward pass
            # -----------------------------

            predictions = model(inputs)

            # -----------------------------
            # Compute loss
            # -----------------------------

            loss = criterion(
            predictions,
            targets
)

            # -----------------------------
            # Accumulate weighted loss
            # -----------------------------

            running_loss += loss.item() * inputs.size(0)

    # ------------------------------------------
    # Average loss
    # ------------------------------------------

    average_loss = (
        running_loss /
        len(validation_loader.dataset)
    )

    return average_loss

def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device
):
    """
    Trains the neural network.
    """

    best_validation_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    train_history = []

    validation_history = []

    for epoch in range(config.EPOCHS):

        # -------------------------------------
        # Training
        # -------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # -------------------------------------
        # Validation
        # -------------------------------------

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        scheduler.step(validation_loss)

        train_history.append(train_loss)
        validation_history.append(validation_loss)

        # -------------------------------------
        # Save best model
        # -------------------------------------

        if validation_loss < (
            best_validation_loss - config.MIN_DELTA
        ):
            best_validation_loss = validation_loss

            best_epoch = epoch + 1

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "learning_rate": config.LEARNING_RATE,
                    "batch_size": config.BATCH_SIZE,
                    "architecture": config.HIDDEN_LAYERS,
                },
                config.MODEL_PATH
            )

            print("✓ Best model updated")

        else:

            epochs_without_improvement += 1

        # -------------------------------------
        # Progress
        # -------------------------------------

        print(

            f"Epoch [{epoch + 1:3d}/{config.EPOCHS}] "

            f"Train Loss: {train_loss:.6f} | "

            f"Validation Loss: {validation_loss:.6f}"

        )

        # -------------------------------------
        # Early stopping
        # -------------------------------------

        if epochs_without_improvement >= config.PATIENCE:

            print()

            print("Early stopping triggered.")

            print(
                f"No improvement for "
                f"{config.PATIENCE} consecutive epochs."
            )

            break

    return {

    "train_loss": train_history,

    "validation_loss": validation_history,

    "best_validation_loss": best_validation_loss,

    "best_epoch": best_epoch

}

def save_training_results(history):
    """
    Saves the training history and loss curves.
    """

    # ------------------------------------------
    # Save CSV
    # ------------------------------------------

    history_df = pd.DataFrame({

        "Epoch": range(
            1,
            len(history["train_loss"]) + 1
        ),

        "Train Loss": history["train_loss"],

        "Validation Loss": history["validation_loss"]

    })

    csv_path = os.path.join(
        config.RESULTS_DIRECTORY,
        "loss_history.csv"
    )

    history_df.to_csv(
        csv_path,
        index=False
    )

    # ------------------------------------------
    # Plot Loss Curves
    # ------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history["validation_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training History")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    figure_path = os.path.join(
        config.RESULTS_DIRECTORY,
        "loss_curve.png"
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    print()

    print("Training history saved.")

    print(f"CSV : {csv_path}")

    print(f"Plot: {figure_path}")

def main():

    set_seed()

    create_directories()

    device = get_device()

    train_loader, validation_loader,input_scaler,output_scaler, = prepare_data()

    # --------------------------------------------------
    # Save scaler
    # --------------------------------------------------
    joblib.dump(
    input_scaler,
    config.SCALER_PATH,
    )

    joblib.dump(
    output_scaler,
    config.OUTPUT_SCALER_PATH,
)

    model, criterion, optimizer, scheduler = build_model(device)

    history = train_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
        device
)

    save_training_results(history)

    print()

    print("=" * 60)

    print("Training Complete")

    print("=" * 60)

    print(
    f"Best Validation Loss : "
    f"{history['best_validation_loss']:.6f}"
    )

    print(
    f"Best Epoch           : "
    f"{history['best_epoch']}"  
    )

    print(f"Model Saved          : {config.MODEL_PATH}")

    print(f"Scaler Saved         : {config.SCALER_PATH}")

    print("=" * 60)


if __name__ == "__main__":

    main()