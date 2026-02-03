import numpy as np
import pandas as pd

kappa_values = np.linspace(0.01, 0.08, 40)
length_values = np.linspace(1, 50, 200)

data = []
for kappa in kappa_values:
    for L in length_values:
        split_ratio = np.sin(kappa * L) ** 2
        data.append([kappa, L, split_ratio])

df = pd.DataFrame(
    data, columns=["kappa", "length_um", "split_ratio"]
)

df.to_csv("data/raw/coupler_dataset.csv", index=False)
print(f"Dataset generated: {len(df)} samples")
