"""
Free Spectral Range calculations.
"""

import numpy as np


def calculate_fsr(resonances):
    """
    Calculate the Free Spectral Range (FSR)
    from detected resonance wavelengths.

    Parameters
    ----------
    resonances : ndarray
        Resonance wavelengths (meters).

    Returns
    -------
    fsr : ndarray
        FSR values (meters).

    average_fsr : float
        Average FSR (meters).
    """

    fsr = np.diff(resonances)

    average_fsr = np.mean(fsr)

    return fsr, average_fsr