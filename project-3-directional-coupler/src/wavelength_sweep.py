import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Nominal design choice
# -----------------------------

kappa_0 = 0.05       # nominal coupling
L = 20.0             # µm (chosen from Pareto front)
lambda_0 = 1550      # nm

# Wavelength sweep
wavelengths = np.linspace(1500, 1600, 200)

# Simple wavelength dependence model
kappa_lambda = kappa_0 * (lambda_0 / wavelengths)

split_ratio = np.sin(kappa_lambda * L) ** 2

# -----------------------------
# Plot wavelength response
# -----------------------------

plt.figure(figsize=(7, 5))
plt.plot(wavelengths, split_ratio)

plt.axhline(0.5, linestyle="--")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Split Ratio")
plt.title("Wavelength Sensitivity of Directional Coupler")

plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/wavelength_response.png", dpi=300)
plt.close()

print("Wavelength sweep completed.")
