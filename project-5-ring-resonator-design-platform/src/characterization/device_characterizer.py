"""
device_characterizer.py

Centralized characterization pipeline for a ring resonator.

This module performs all calculations required to characterize
a simulated ring resonator and returns the results in a
single dictionary.
"""

import numpy as np

from src.physics.simulator import simulate

from src.analysis.peak_detection import find_resonances
from src.analysis.fsr import calculate_fsr
from src.analysis.fwhm import calculate_fwhm
from src.analysis.q_factor import calculate_q
from src.analysis.extinction_ratio import calculate_extinction_ratio
from src.analysis.insertion_loss import calculate_insertion_loss
from src.analysis.finesse import calculate_finesse


def characterize_device(
    wavelengths,
    radius,
    n_eff,
    t,
    a
):
    """
    Characterize a single ring resonator.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength sweep (m)

    radius : float
        Ring radius (m)

    n_eff : float
        Effective index

    t : float
        Self-coupling coefficient

    a : float
        Round-trip amplitude transmission

    Returns
    -------
    dict
        Dictionary containing all characterization results.
    """

    # ==========================================================
    # Run Simulation
    # ==========================================================

    transmission = simulate(
        wavelengths,
        radius,
        n_eff,
        t,
        a
    )

    # ==========================================================
    # Detect Resonances
    # ==========================================================

    resonances, indices = find_resonances(
        wavelengths,
        transmission
    )

    if len(resonances) < 2:

        return {
            "valid": False
        }

    # ==========================================================
    # Free Spectral Range
    # ==========================================================

    fsr, average_fsr = calculate_fsr(
        resonances
    )

    # ==========================================================
    # FWHM
    # ==========================================================

    fwhm = calculate_fwhm(
        wavelengths,
        transmission,
        indices
    )

    average_fwhm = np.mean(fwhm)

    # ==========================================================
    # Loaded Q
    # ==========================================================

    q = calculate_q(
        resonances,
        fwhm
    )

    average_q = np.mean(q)

    # ==========================================================
    # Extinction Ratio
    # ==========================================================

    er = calculate_extinction_ratio(
        transmission,
        indices
    )

    average_er = np.mean(er)

    # ==========================================================
    # Insertion Loss
    # ==========================================================

    insertion_loss = calculate_insertion_loss(
        transmission
    )

    # ==========================================================
    # Finesse
    # ==========================================================

    finesse = calculate_finesse(
        fsr,
        fwhm[:-1]
    )

    average_finesse = np.mean(finesse)

     # ==========================================================
    # Build Resonance Database
    # ==========================================================

    resonance_data = []

    number_of_resonances = len(resonances)

    for i in range(number_of_resonances):

        resonance = {

             "id": i + 1,

            "peak_index": int(indices[i]),

            "wavelength": float(resonances[i]),

            "fwhm": float(fwhm[i]),

            "q": float(q[i]),

            "er": float(er[i])

        }

        if i < len(fsr):

            resonance["fsr"] = float(fsr[i])

        else:

            resonance["fsr"] = np.nan

        if i < len(finesse):

            resonance["finesse"] = float(finesse[i])

        else:

            resonance["finesse"] = np.nan

        resonance_data.append(resonance)

    # ==========================================================
    # Return Everything
    # ==========================================================

    return {

        "valid": True,

        "radius": radius,
        "n_eff": n_eff,
        "t": t,
        "a": a,

        "wavelengths": wavelengths,
        "transmission": transmission,

        "resonances": resonances,
        "indices": indices,

        "number_of_resonances": number_of_resonances,

        "resonance_data": resonance_data,

        "fsr": fsr,
        "average_fsr": average_fsr,

        "fwhm": fwhm,
        "average_fwhm": average_fwhm,

        "q": q,
        "average_q": average_q,

        "er": er,
        "average_er": average_er,

        "insertion_loss": insertion_loss,

        "finesse": finesse,
        "average_finesse": average_finesse
    }