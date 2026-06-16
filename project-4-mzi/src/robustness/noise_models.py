import numpy as np


def add_phase_noise(delta_phi, sigma=0.05):
    """
    Gaussian phase noise.
    """
    noise = np.random.normal(0, sigma)
    return delta_phi + noise


def fabrication_variation(length, sigma=0.5e-6):
    """
    Simulate fabrication length variation.
    """
    variation = np.random.normal(0, sigma)
    return length + variation


def wavelength_shift(wavelength, sigma=1e-9):
    """
    Simulate wavelength drift.
    """
    shift = np.random.normal(0, sigma)
    return wavelength + shift