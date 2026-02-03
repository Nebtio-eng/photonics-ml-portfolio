# src/ring_simulation.py

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.signal import find_peaks

# -------------------------------
# Parameters
# -------------------------------
resolution = 20
cell = mp.Vector3(16, 16)
pml_layers = [mp.PML(1.0)]

waveguide_width = 0.5
ring_radius = 15.0      # CHANGE FREELY
ring_width = 0.5
gap = 0.2

ring_shift = ring_radius + waveguide_width/2 + gap

fcen = 1 / 1.55
df = 0.3
nfreq = 400

RESULTS_DIR = "results/spectra"
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------------
# Source
# -------------------------------
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(-6, 0)
    )
]

# -------------------------------
# Simulation function
# -------------------------------
def run_simulation(geometry):
    if hasattr(mp, "reset_meep"):
        mp.reset_meep()

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution
    )

    tran = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(6, 0), size=mp.Vector3(0, 2))
    )

    sim.run(until=5000)

    flux = np.array(mp.get_fluxes(tran))
    freqs = np.array(mp.get_flux_freqs(tran))
    wl = 1 / freqs

    return wl, flux, freqs

# -------------------------------
# Geometries
# -------------------------------
geometry_ref = [
    mp.Block(
        size=mp.Vector3(mp.inf, waveguide_width, mp.inf),
        center=mp.Vector3(0, 0),
        material=mp.Medium(index=3.48)
    )
]

geometry_ring = geometry_ref + [
    mp.Cylinder(
        radius=ring_radius,
        height=mp.inf,
        center=mp.Vector3(0, ring_shift),
        material=mp.Medium(index=3.48)
    ),
    mp.Cylinder(
        radius=ring_radius - ring_width,
        height=mp.inf,
        center=mp.Vector3(0, ring_shift),
        material=mp.air
    )
]

# -------------------------------
# Run simulations
# -------------------------------
wl_ref, flux_ref, freqs = run_simulation(geometry_ref)
wl_ring, flux_ring, _ = run_simulation(geometry_ring)

T = flux_ring / flux_ref

# -------------------------------
# FSR & Q
# -------------------------------
inv_T = 1 - T
peaks, _ = find_peaks(inv_T, height=0.05)

res_wl = wl_ref[peaks]
fsr = np.mean(np.abs(np.diff(res_wl)))

dip = peaks[0]
half = (1 + T[dip]) / 2
l, r = dip, dip
while T[l] < half and l > 0: l -= 1
while T[r] < half and r < len(T)-1: r += 1
Q = res_wl[0] / abs(wl_ref[r] - wl_ref[l])

# -------------------------------
# Plot
# -------------------------------
plt.plot(wl_ref, T)
plt.gca().invert_xaxis()
plt.xlabel("Wavelength (µm)")
plt.ylabel("Normalized Transmission")
plt.title("Ring Resonator Spectrum")
plt.savefig(f"{RESULTS_DIR}/spectrum_{timestamp}.png", dpi=300)
plt.show()

print(f"FSR ≈ {fsr:.4f} µm | Q ≈ {Q:.2f}")


def run_ring_and_extract_features(ring_radius, gap):
    """
    Runs ring resonator simulation for given geometry
    and returns extracted features for ML.
    """

    # --- compute ring shift automatically ---
    ring_shift = ring_radius + waveguide_width/2 + gap

    # --- redefine geometry ---
    geometry_ring = geometry_ref + [
        mp.Cylinder(
            radius=ring_radius,
            height=mp.inf,
            center=mp.Vector3(0, ring_shift),
            material=mp.Medium(index=3.48)
        ),
        mp.Cylinder(
            radius=ring_radius - ring_width,
            height=mp.inf,
            center=mp.Vector3(0, ring_shift),
            material=mp.air
        )
    ]

    # --- run simulations ---
    wl_ref, flux_ref, _ = run_simulation(geometry_ref)
    wl_ring, flux_ring, _ = run_simulation(geometry_ring)

    T = flux_ring / flux_ref

    # --- extract dips ---
    inv_T = 1 - T
    peaks, _ = find_peaks(inv_T, height=0.05)

    res_wl = wl_ref[peaks]
    fsr = np.mean(np.abs(np.diff(res_wl)))

    # --- Q-factor ---
    dip = peaks[0]
    half = (1 + T[dip]) / 2
    l, r = dip, dip
    while T[l] < half and l > 0: l -= 1
    while T[r] < half and r < len(T)-1: r += 1
    Q = res_wl[0] / abs(wl_ref[r] - wl_ref[l])

    dip_depth = np.min(T)

    return fsr, Q, dip_depth
