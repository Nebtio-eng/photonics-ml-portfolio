"""
Full Width at Half Maximum calculation.
"""

import numpy as np


def interpolate(x1, y1, x2, y2, y_target):
    """
    Linear interpolation.
    """
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)


def calculate_fwhm(
    wavelengths,
    transmission,
    indices
):
    """
    Calculate FWHM for every resonance.
    """

    fwhm_values = []

    for idx in indices:

        Tmin = transmission[idx]
        Tmax = np.max(transmission)

        Thalf = Tmin + (Tmax - Tmin) / 2

        left = idx
        while left > 0 and transmission[left] < Thalf:
            left -= 1

        right = idx
        while right < len(transmission)-1 and transmission[right] < Thalf:
            right += 1

        lambda_left = interpolate(
            wavelengths[left],
            transmission[left],
            wavelengths[left+1],
            transmission[left+1],
            Thalf
        )

        lambda_right = interpolate(
            wavelengths[right-1],
            transmission[right-1],
            wavelengths[right],
            transmission[right],
            Thalf
        )

        fwhm = lambda_right - lambda_left

        fwhm_values.append(fwhm)

    return np.array(fwhm_values)