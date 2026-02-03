# src/coupling_sweep.py

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results/coupling", exist_ok=True)

gaps = [0.1, 0.2, 0.3, 0.4]
dip_depths = [0.85, 0.45, 0.70, 0.90]  # replace with measured values

plt.plot(gaps, dip_depths, marker='o')
plt.xlabel("Coupling gap (µm)")
plt.ylabel("Minimum transmission")
plt.title("Critical Coupling Study")
plt.grid()
plt.savefig("results/coupling/critical_coupling.png", dpi=300)
plt.show()
