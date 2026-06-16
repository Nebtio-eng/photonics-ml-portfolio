import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.physics.phase_model import (
    phase_difference
)

from src.physics.mzi_transfer import (
    mzi_transmission
)


def transmission_vs_deltaL():

    wavelength = 1550e-9
    neff = 2.4

    L1 = 500e-6

    delta_L_values = np.linspace(
        -20e-6,
        20e-6,
        1000
    )

    transmissions = []

    for delta_L in delta_L_values:

        L2 = L1 + delta_L

        delta_phi = phase_difference(
            neff,
            L1,
            neff,
            L2,
            wavelength
        )

        T = mzi_transmission(delta_phi)

        transmissions.append(T)

    plt.figure(figsize=(8,6))

    plt.plot(
        delta_L_values * 1e6,
        transmissions
    )

    plt.xlabel("ΔL (µm)")
    plt.ylabel("Transmission")

    plt.title(
        "MZI Transmission vs Arm Length Difference"
    )

    plt.grid(True)

    plt.savefig(
        "results/transmission_vs_deltaL.png"
    )

    plt.close()


if __name__ == "__main__":
    transmission_vs_deltaL()