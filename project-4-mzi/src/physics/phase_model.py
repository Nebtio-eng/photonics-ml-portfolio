import numpy as np

def propagation_constant(neff, wavelength):
    """
    Compute propagation constant beta.
    """
    return 2 * np.pi * neff / wavelength


def phase_accumulation(neff, length, wavelength):
    """
    Compute phase accumulation.
    """
    beta = propagation_constant(neff, wavelength)
    return beta * length


def phase_difference(neff1, L1, neff2, L2, wavelength):
    """
    Compute phase difference between arms.
    """
    phi1 = phase_accumulation(neff1, L1, wavelength)
    phi2 = phase_accumulation(neff2, L2, wavelength)

    return phi1 - phi2
