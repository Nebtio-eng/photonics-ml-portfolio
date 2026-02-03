# src/fsr_ng_analysis.py

import numpy as np

# Example: resonance frequencies from MEEP (1/µm)
res_freqs_meep = np.array([
    0.645, 0.662, 0.679  # REPLACE with your extracted values
])

# Ring radius
ring_radius_um = 15.0

# --- Convert to SI units ---
c = 3e8  # m/s
res_freqs_hz = res_freqs_meep * (c / 1e-6)

# --- Frequency FSR ---
df_hz = np.mean(np.abs(np.diff(res_freqs_hz)))

# --- Group index ---
R_m = ring_radius_um * 1e-6
ng = c / (df_hz * 2 * np.pi * R_m)

print(f"Estimated group index ng ≈ {ng:.2f}")
