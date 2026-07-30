"""
Finesse calculations.
"""

import numpy as np


def calculate_finesse(
    fsr,
    fwhm
):
    """
    Calculate finesse values.

    Parameters
    ----------
    fsr : ndarray
        Free Spectral Range values.

    fwhm : ndarray
        Full Width at Half Maximum values.

    Returns
    -------
    finesse : ndarray
        Finesse values.
    """

    finesse = fsr / fwhm

    return finesse