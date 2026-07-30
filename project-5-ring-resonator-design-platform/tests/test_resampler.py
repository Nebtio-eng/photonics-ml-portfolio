"""
Verify that the resampled spectrum preserves the original spectrum.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.physics.simulator import simulate
from src.dataset.spectrum_resampler import resample_spectrum


# --------------------------------------------------
# Simulation Parameters
# --------------------------------------------------

wavelengths = np.linspace(
    1500e-9,
    1600e-9,
    5000
)

radius = 10e-6
n_eff = 2.40
t = 0.92
a = 0.99

# --------------------------------------------------
# Original Spectrum
# --------------------------------------------------

original = simulate(
    wavelengths=wavelengths,
    radius=radius,
    n_eff=n_eff,
    t=t,
    a=a
)

# --------------------------------------------------
# Resampled Spectrum
# --------------------------------------------------

new_wavelengths, new_spectrum = resample_spectrum(
    wavelengths,
    original,
    num_points=1024
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    wavelengths * 1e9,
    original,
    label="Original (5000)"
)

plt.scatter(
    new_wavelengths * 1e9,
    new_spectrum,
    s=12,
    color="red",
    label="Resampled (1024)"
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
plt.title("Spectrum Resampling Verification")

plt.grid(True)
plt.legend()

plt.show()