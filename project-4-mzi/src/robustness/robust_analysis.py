import numpy as np

from src.physics.phase_model import phase_difference
from src.physics.mzi_transfer import mzi_transmission

from src.robustness.noise_models import (
    add_phase_noise,
    fabrication_variation,
    wavelength_shift
)


def robust_transmission(
    neff,
    L1,
    L2,
    wavelength,
    n_trials=100
):
    """
    Monte Carlo robustness analysis.
    """

    transmissions = []

    for _ in range(n_trials):

        noisy_L1 = fabrication_variation(L1)
        noisy_L2 = fabrication_variation(L2)

        noisy_wavelength = wavelength_shift(wavelength)

        delta_phi = phase_difference(
            neff,
            noisy_L1,
            neff,
            noisy_L2,
            noisy_wavelength
        )

        delta_phi = add_phase_noise(delta_phi)

        T = mzi_transmission(delta_phi)

        transmissions.append(T)

    return {
        "mean_transmission": np.mean(transmissions),
        "std_transmission": np.std(transmissions),
        "min_transmission": np.min(transmissions),
        "max_transmission": np.max(transmissions)
    }