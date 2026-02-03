import csv
import os
from ring_simulation import run_ring_and_extract_features

# -----------------------------
# Parameter sweep
# -----------------------------
ring_radii = [10.0, 12.0, 15.0]     # µm
gaps = [0.15, 0.20, 0.25]           # µm

# -----------------------------
# Output file
# -----------------------------
os.makedirs("results", exist_ok=True)
csv_file = "results/ring_dataset.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ring_radius", "gap", "FSR", "Q_factor", "dip_depth"])

    for R in ring_radii:
        for g in gaps:
            print(f"Running simulation: R={R} µm, gap={g} µm")

            fsr, Q, dip = run_ring_and_extract_features(R, g)

            writer.writerow([R, g, fsr, Q, dip])

print(f"\nDataset saved to {csv_file}")
