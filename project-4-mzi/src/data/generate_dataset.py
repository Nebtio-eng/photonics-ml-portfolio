import numpy as np
import pandas as pd

from src.physics.phase_model import phase_difference
from src.physics.mzi_transfer import mzi_transmission


def generate_dataset(n_samples=50000):
    wavelength = 1550e-9

    data = []

    for _ in range(n_samples):
        L1 = np.random.uniform(400e-6, 600e-6)
        delta_L = np.random.uniform(-20e-6, 20e-6)

        L2 = L1 + delta_L

        neff = np.random.uniform(2.3, 2.5)

        delta_phi = phase_difference(
            neff,
            L1,
            neff,
            L2,
            wavelength
        )

        T = mzi_transmission(delta_phi)

        data.append([
            L1,
            L2,
            delta_L,
            neff,
            wavelength,
            T
        ])

    columns = [
        "L1",
        "L2",
        "delta_L",
        "neff",
        "wavelength",
        "transmission"
    ]

    df = pd.DataFrame(data, columns=columns)

    df.to_csv("results/mzi_dataset.csv", index=False)

    print("Dataset saved to results/mzi_dataset.csv")


if __name__ == "__main__":
    generate_dataset()

