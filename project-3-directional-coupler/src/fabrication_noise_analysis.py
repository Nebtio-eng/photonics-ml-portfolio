import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("data/raw/coupler_dataset.csv")

# Target 50:50
target = 0.5

# Noise level (± percentage)
noise_level = 0.1  # 10% fabrication variation

# -----------------------------
# Apply fabrication noise
# -----------------------------

np.random.seed(42)

df["kappa_noisy"] = df["kappa"] * (
    1 + noise_level * np.random.randn(len(df))
)

df["split_noisy"] = np.sin(df["kappa_noisy"] * df["length_um"]) ** 2

df["error_clean"] = np.abs(df["split_ratio"] - target)
df["error_noisy"] = np.abs(df["split_noisy"] - target)

# -----------------------------
# Plot robustness comparison
# -----------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    df["kappa"],
    df["length_um"],
    c=df["error_noisy"],
    cmap="inferno",
    s=10
)

plt.xlabel("Coupling Coefficient κ (gap proxy)")
plt.ylabel("Interaction Length (µm)")
plt.title("Effect of Fabrication Noise on 50:50 Directional Coupler")
plt.colorbar(label="Split Error with Fabrication Noise")

plt.tight_layout()
plt.savefig("results/figures/fabrication_noise_effect.png", dpi=300)
plt.close()

print("Fabrication noise analysis completed.")
