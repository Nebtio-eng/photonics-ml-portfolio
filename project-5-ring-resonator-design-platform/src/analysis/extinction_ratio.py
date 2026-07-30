"""
Extinction Ratio calculations.
"""

import numpy as np


def calculate_extinction_ratio(
    transmission,
    indices
):
    """
    Calculate extinction ratio for each resonance.

    Parameters
    ----------
    transmission : ndarray
        Transmission spectrum.

    indices : ndarray
        Resonance indices.

    Returns
    -------
    er : ndarray
        Extinction ratios in dB.
    """

    Tmax = np.max(transmission)

    Tmin = transmission[indices]

    er = 10 * np.log10(
        Tmax / Tmin
    )

    return er