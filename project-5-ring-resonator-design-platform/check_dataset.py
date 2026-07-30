import pandas as pd

df = pd.read_csv("data/ring_resonator_dataset.csv")

print("=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("Missing Values")
print("=" * 60)
print(df.isnull().sum().sum())

print("\n")

print("=" * 60)
print("Duplicate Rows")
print("=" * 60)
print(df.duplicated().sum())

print("\n")

print("=" * 60)
print("Statistics")
print("=" * 60)
print(df.describe())