"""
Resonance-related calculations.
"""

import numpy as np


def propagation_constant(
    wavelength: float,
    n_eff: float
) -> float:
    """
    Calculate propagation constant.
    """
    return 2 * np.pi * n_eff / wavelength


def round_trip_phase(
    beta: float,
    circumference: float
) -> float:
    """
    Calculate round-trip phase.
    """
    return beta * circumference