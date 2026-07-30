"""
Automatic resonance detection.
"""

import numpy as np
from scipy.signal import find_peaks


def find_resonances(
    wavelengths,
    transmission,
    prominence=0.05
):
    """
    Detect resonance wavelengths.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array.

    transmission : ndarray
        Transmission spectrum.

    prominence : float
        Minimum peak prominence.

    Returns
    -------
    resonance_wavelengths
    resonance_indices
    """

    inverted = 1 - transmission

    peaks, properties = find_peaks(
        inverted,
        prominence=prominence
    )

    return wavelengths[peaks], peaks
