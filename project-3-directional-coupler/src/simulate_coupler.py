import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

L = np.linspace(0, 50, 1000)
kappa_values = [0.02, 0.04, 0.06]

plt.figure(figsize=(8, 5))
for kappa in kappa_values:
    P2 = np.sin(kappa * L) ** 2
    plt.plot(L, P2, label=f"κ = {kappa}")

plt.xlabel("Interaction Length (µm)")
plt.ylabel("Power in Waveguide 2")
plt.title("Directional Coupler Power Transfer")
plt.legend()
plt.grid(True)
plt.savefig("results/figures/power_vs_length.png", dpi=300)
plt.close()
