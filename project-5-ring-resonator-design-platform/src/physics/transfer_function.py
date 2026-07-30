"""
Ring resonator transmission model.
"""

import numpy as np


def transmission(
    phase: float,
    t: float,
    a: float
) -> float:
    """
    Calculate through-port transmission.
    """
    numerator = (
        t**2
        + a**2
        - 2 * a * t * np.cos(phase)
    )

    denominator = (
        1
        + (a * t) ** 2
        - 2 * a * t * np.cos(phase)
    )

    return numerator / denominator

