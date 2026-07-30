"""
Insertion Loss calculations.
"""

import numpy as np


def calculate_insertion_loss(transmission):
    """
    Calculate insertion loss from the maximum transmission.

    Parameters
    ----------
    transmission : ndarray
        Normalized transmission spectrum.

    Returns
    -------
    insertion_loss : float
        Insertion loss (dB).
    """

    Tmax = np.max(transmission)

    insertion_loss = -10 * np.log10(Tmax)

    return insertion_loss