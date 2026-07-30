"""
spectrum_resampler.py

Utility functions for reducing the transmission spectrum
to a fixed number of wavelength samples for machine learning.
"""

import numpy as np


def resample_spectrum(
    wavelengths,
    transmission,
    num_points=1024
):
    """
    Uniformly resample a transmission spectrum.

    Parameters
    ----------
    wavelengths : ndarray
        Original wavelength sweep.

    transmission : ndarray
        Original transmission spectrum.

    num_points : int
        Number of output wavelength samples.

    Returns
    -------
    tuple(ndarray, ndarray)

        resampled_wavelengths
        resampled_transmission
    """

    # New wavelength grid

    resampled_wavelengths = np.linspace(
        wavelengths[0],
        wavelengths[-1],
        num_points
    )

    # Linear interpolation

    resampled_transmission = np.interp(
        resampled_wavelengths,
        wavelengths,
        transmission
    )

    return (
        resampled_wavelengths,
        resampled_transmission
    )