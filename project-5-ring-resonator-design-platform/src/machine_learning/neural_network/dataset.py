import os
import joblib

import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class RingResonatorDataset(Dataset):
    """
    PyTorch Dataset for the Ring Resonator surrogate model.

    Responsibilities:
    - Load the dataset from CSV
    - Split inputs and outputs
    - Normalize the input features
    - Save or load the input scaler
    - Convert everything into PyTorch tensors
    """

    def __init__(
        self,
        csv_file,
        scaler_path=None,
        fit_scaler=False
    ):

        # --------------------------------------------------
        # Load CSV
        # --------------------------------------------------
        df = pd.read_csv(csv_file)

        # --------------------------------------------------
        # Split inputs and outputs
        # --------------------------------------------------
        inputs = df.iloc[:, :4].values
        outputs = df.iloc[:, 4:].values

        # --------------------------------------------------
        # Normalize inputs
        # --------------------------------------------------
        if fit_scaler:

            self.scaler = StandardScaler()

            inputs = self.scaler.fit_transform(inputs)

            if scaler_path is not None:

                os.makedirs(
                    os.path.dirname(scaler_path),
                    exist_ok=True
                )

                joblib.dump(
                    self.scaler,
                    scaler_path
                )

        else:

            if scaler_path is None:
                raise ValueError(
                    "A scaler_path must be provided when fit_scaler=False."
                )

            self.scaler = joblib.load(scaler_path)

            inputs = self.scaler.transform(inputs)

        # --------------------------------------------------
        # Convert NumPy arrays to PyTorch tensors
        # --------------------------------------------------
        self.inputs = torch.tensor(
            inputs,
            dtype=torch.float32
        )

        self.outputs = torch.tensor(
            outputs,
            dtype=torch.float32
        )

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Returns one training sample.
        """
        return (
            self.inputs[index],
            self.outputs[index]
        )