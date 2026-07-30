"""
Ring geometry calculations.
"""

from .constants import PI


def circumference(radius: float) -> float:
    """
    Calculate the circumference of a ring.

    Parameters
    ----------
    radius : float
        Ring radius in meters.

    Returns
    -------
    float
        Circumference in meters.
    """
    return 2 * PI * radius
