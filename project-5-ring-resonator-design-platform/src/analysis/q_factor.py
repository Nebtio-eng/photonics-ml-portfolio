"""
Quality Factor calculations.
"""

import numpy as np


def calculate_q(
    resonances,
    fwhm
):
    """
    Calculate loaded Q-factor.

    Parameters
    ----------
    resonances : ndarray
        Resonance wavelengths.

    fwhm : ndarray
        Full Width at Half Maximum.

    Returns
    -------
    q : ndarray
        Quality factors.
    """

    q = resonances / fwhm

    return q