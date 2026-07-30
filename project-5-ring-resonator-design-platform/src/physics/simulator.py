import numpy as np

from .geometry import circumference
from .resonance import propagation_constant
from .resonance import round_trip_phase
from .transfer_function import transmission


def simulate(
    wavelengths,
    radius,
    n_eff,
    t,
    a
):
    """
    Generate transmission spectrum.
    """

    L = circumference(radius)

    spectrum = []

    for wavelength in wavelengths:

        beta = propagation_constant(
            wavelength,
            n_eff
        )

        phase = round_trip_phase(
            beta,
            L
        )

        T = transmission(
            phase,
            t,
            a
        )

        spectrum.append(T)

    return np.array(spectrum)