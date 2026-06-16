import numpy as np

def mzi_transmission(delta_phi):
    """
    Ideal MZI transmission.
    """
    return np.cos(delta_phi / 2) ** 2


def extinction_ratio(transmission):
    """
    Compute extinction ratio.
    """
    Tmax = np.max(transmission)
    Tmin = np.min(transmission)

    return Tmax / Tmin

