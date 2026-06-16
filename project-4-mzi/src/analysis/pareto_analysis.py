import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.robustness.robust_analysis import robust_transmission


def compute_sensitivity(L1, L2):
    """
    Simple sensitivity metric.
    Larger delta_L => higher sensitivity.
    """
    return abs(L2 - L1)


def run_pareto_analysis():

    designs = []

    for _ in range(200):

        L1 = np.random.uniform(400e-6, 600e-6)

        delta_L = np.random.uniform(1e-6, 20e-6)

        L2 = L1 + delta_L

        robustness = robust_transmission(
            neff=2.4,
            L1=L1,
            L2=L2,
            wavelength=1550e-9
        )

        sensitivity = compute_sensitivity(L1, L2)

        robustness_score = 1 / (
            robustness["std_transmission"] + 1e-6
        )

        designs.append([
            sensitivity,
            robustness_score
        ])

    designs = np.array(designs)

    plt.figure(figsize=(8,6))

    plt.scatter(
        designs[:,0] * 1e6,
        designs[:,1]
    )

    plt.xlabel("Sensitivity (ΔL in µm)")
    plt.ylabel("Robustness Score")

    plt.title("Pareto Tradeoff: Sensitivity vs Robustness")

    plt.savefig(
        "results/pareto_front.png"
    )

    plt.close()


if __name__ == "__main__":
    run_pareto_analysis()