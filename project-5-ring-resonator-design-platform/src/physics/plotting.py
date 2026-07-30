import matplotlib.pyplot as plt


def plot_spectrum(
    wavelengths,
    transmission,
    resonances=None,
    indices=None
):
    """
    Plot the transmission spectrum.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(
        wavelengths * 1e9,
        transmission,
        linewidth=2,
        label="Transmission"
    )

    # Plot resonance markers if provided
    if resonances is not None and indices is not None:

        plt.scatter(
            resonances * 1e9,
            transmission[indices],
            color="red",
            s=40,
            zorder=5,
            label="Detected Resonances"
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("Ring Resonator Transmission Spectrum")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()