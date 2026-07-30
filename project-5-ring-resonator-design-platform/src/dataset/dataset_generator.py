"""
dataset_generator.py

Generate a dataset of ring resonator devices for
machine learning and design exploration.
"""

import csv
import os

import numpy as np

from src.physics.simulator import simulate
from scipy.stats import qmc
from src.dataset.spectrum_resampler import resample_spectrum


HEADER = [

    "Radius_um",
    "Effective_Index",
    "Self_Coupling_t",
    "Loss_Factor_a"

]

for i in range(1024):

    HEADER.append(f"Transmission_{i:03d}")


def create_data_directory():

    os.makedirs("data", exist_ok=True)


def initialize_dataset(path):

    with open(path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(HEADER)

def append_device(
    path,
    radius,
    n_eff,
    t,
    a,
    wavelengths,
    transmission
):

    _, transmission = resample_spectrum(
        wavelengths,
        transmission,
        num_points=1024
    )

    writer_row = [

        radius * 1e6,
        n_eff,
        t,
        a

    ]

    writer_row.extend(transmission.tolist())

    with open(path, "a", newline="") as file:

        writer = csv.writer(file)
        writer.writerow(writer_row)


def generate_dataset():

    print("\nGenerating Latin Hypercube dataset...\n")

    create_data_directory()

    dataset_path = "data/ring_resonator_dataset.csv"

    initialize_dataset(dataset_path)

    wavelengths = np.linspace(
        1500e-9,
        1600e-9,
        5000
    )

    # Save wavelength grid once
    resampled_wavelengths, _ = resample_spectrum(
        wavelengths,
        np.zeros_like(wavelengths),
        num_points=1024
    )

    np.save(
        "data/wavelength_grid.npy",
        resampled_wavelengths
    )

    # ==========================================================
    # Number of Devices
    # ==========================================================

    num_samples = 10000

    # ==========================================================
    # Latin Hypercube Sampler
    # ==========================================================

    sampler = qmc.LatinHypercube(
        d=4,
        seed=42
    )

    samples = sampler.random(
        n=num_samples
    )

    # ==========================================================
    # Scale Samples
    # ==========================================================

    lower_bounds = [
        5e-6,
        2.20,
        0.80,
        0.94
    ]

    upper_bounds = [
        30e-6,
        2.60,
        0.99,
        0.999
    ]

    samples = qmc.scale(
        samples,
        lower_bounds,
        upper_bounds
    )

    # ==========================================================
    # Generate Dataset
    # ==========================================================

    for i, sample in enumerate(samples):

        radius = sample[0]
        n_eff = sample[1]
        t = sample[2]
        a = sample[3]

        transmission = simulate(

            wavelengths=wavelengths,

            radius=radius,

            n_eff=n_eff,

            t=t,

            a=a

        )

        append_device(

            dataset_path,

            radius,
            n_eff,
            t,
            a,

            wavelengths,
            transmission

        )

        if (i + 1) % 50 == 0:

            print(
                f"{i + 1}/{num_samples} devices completed"
            )

    print("\nDataset generation complete.")
    print(f"Saved to: {dataset_path}")

    