"""
Project 5 - Ring Resonator Design Platform

Main simulation and characterization pipeline.

Current Features

- Analytical ring resonator simulation
- Device characterization pipeline
- Automatic report generation
- Automatic dataset generation
- Transmission spectrum visualization
- Free Spectral Range (FSR)
- Full Width at Half Maximum (FWHM)
- Loaded Quality Factor (Q)
- Extinction Ratio (ER)
- Insertion Loss (IL)
- Finesse
"""

import numpy as np

# =============================================================================
# Physics Modules
# =============================================================================


from src.physics.plotting import plot_spectrum

# =============================================================================
# Analysis Modules
# =============================================================================

from src.characterization.device_characterizer import characterize_device
from src.report.report import generate_report
from src.dataset.dataset_generator import generate_dataset



# =============================================================================
# Ring Parameters
# =============================================================================

radius = 10e-6          # Ring radius (m)
n_eff = 2.40            # Effective index
t = 0.90                # Self-coupling coefficient
a = 0.98                # Round-trip amplitude transmission

# =============================================================================
# Wavelength Sweep
# =============================================================================

wavelengths = np.linspace(
    1500e-9,
    1600e-9,
    5000
)

# =============================================================================
# Characterize Device
# =============================================================================

metrics = characterize_device(
    wavelengths=wavelengths,
    radius=radius,
    n_eff=n_eff,
    t=t,
    a=a
)

if not metrics["valid"]:
    raise RuntimeError(
        "Device characterization failed."
    )

# =============================================================================
# Resonances
# =============================================================================

print("\n" + "=" * 60)
print("DETECTED RESONANCES")
print("=" * 60)

for i, wavelength in enumerate(
    metrics["resonances"],
    start=1
):
    print(
        f"Resonance {i}: "
        f"{wavelength * 1e9:.3f} nm"
    )

# =============================================================================
# FSR
# =============================================================================

print("\n" + "=" * 60)
print("FREE SPECTRAL RANGE (FSR)")
print("=" * 60)

for i, value in enumerate(
    metrics["fsr"],
    start=1
):
    print(
        f"FSR {i}: "
        f"{value * 1e9:.6f} nm"
    )

print(
    f"\nAverage FSR: "
    f"{metrics['average_fsr'] * 1e9:.6f} nm"
)

# =============================================================================
# FWHM
# =============================================================================

print("\n" + "=" * 60)
print("FULL WIDTH AT HALF MAXIMUM (FWHM)")
print("=" * 60)

for i, value in enumerate(
    metrics["fwhm"],
    start=1
):
    print(
        f"FWHM {i}: "
        f"{value * 1e9:.6f} nm"
    )

print(
    f"\nAverage FWHM: "
    f"{metrics['average_fwhm'] * 1e9:.6f} nm"
)

# =============================================================================
# Q
# =============================================================================

print("\n" + "=" * 60)
print("LOADED QUALITY FACTOR (Q)")
print("=" * 60)

for i, value in enumerate(
    metrics["q"],
    start=1
):
    print(f"Q {i}: {value:.0f}")

print(
    f"\nAverage Q: "
    f"{metrics['average_q']:.0f}"
)

# =============================================================================
# ER
# =============================================================================

print("\n" + "=" * 60)
print("EXTINCTION RATIO (ER)")
print("=" * 60)

for i, value in enumerate(
    metrics["er"],
    start=1
):
    print(
        f"ER {i}: {value:.2f} dB"
    )

print(
    f"\nAverage ER: "
    f"{metrics['average_er']:.2f} dB"
)

# =============================================================================
# IL
# =============================================================================

print("\n" + "=" * 60)
print("INSERTION LOSS (IL)")
print("=" * 60)

print(
    f"Insertion Loss: "
    f"{metrics['insertion_loss']:.4f} dB"
)

# =============================================================================
# Finesse
# =============================================================================

print("\n" + "=" * 60)
print("FINESSE")
print("=" * 60)

for i, value in enumerate(
    metrics["finesse"],
    start=1
):
    print(
        f"Finesse {i}: {value:.2f}"
    )

print(
    f"\nAverage Finesse: "
    f"{metrics['average_finesse']:.2f}"
)

# =============================================================================
# Report
# =============================================================================

generate_report(
    radius=metrics["radius"],
    n_eff=metrics["n_eff"],
    t=metrics["t"],
    a=metrics["a"],
    resonances=metrics["resonances"],
    average_fsr=metrics["average_fsr"],
    average_fwhm=metrics["average_fwhm"],
    average_q=metrics["average_q"],
    average_er=metrics["average_er"],
    insertion_loss=metrics["insertion_loss"],
    average_finesse=metrics["average_finesse"]
)

# =============================================================================
# Plot
# =============================================================================

plot_spectrum(
    metrics["wavelengths"],
    metrics["transmission"],
    metrics["resonances"],
    metrics["indices"]
)

# =============================================================================
# Dataset
# =============================================================================

generate_dataset()



