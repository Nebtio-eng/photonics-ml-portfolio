"""
report.py

Automatic characterization report generation for the
Ring Resonator Design Platform.
"""

from pathlib import Path
import csv


def create_results_directory():
    """
    Create the results directory if it does not already exist.

    Returns
    -------
    Path
        Path object pointing to the results directory.
    """

    results_path = Path("results")
    results_path.mkdir(exist_ok=True)

    return results_path


def save_text_report(results_path, report_text):
    """
    Save the report as a text file.

    Parameters
    ----------
    results_path : Path
        Directory where the report will be saved.

    report_text : str
        Formatted report.
    """

    report_file = results_path / "characterization_report.txt"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_text)


def save_csv_report(results_path, header, rows):
    """
    Save the report as CSV.

    Parameters
    ----------
    results_path : Path

    header : list

    rows : list
    """

    csv_file = results_path / "characterization_report.csv"

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)
        writer.writerows(rows)


def generate_report(
    radius,
    n_eff,
    t,
    a,
    resonances,
    average_fsr,
    average_fwhm,
    average_q,
    average_er,
    insertion_loss,
    average_finesse
):
    """
    Generate characterization reports.

    Creates both

    - characterization_report.txt
    - characterization_report.csv
    """

    results_path = create_results_directory()

    # ==========================================================
    # Build Text Report
    # ==========================================================

    report = []
    report.append("=" * 60)
    report.append("RING RESONATOR CHARACTERIZATION REPORT")
    report.append("=" * 60)
    report.append("")

    report.append("DEVICE PARAMETERS")
    report.append("-" * 60)
    report.append(f"Ring Radius          : {radius * 1e6:.3f} µm")
    report.append(f"Effective Index      : {n_eff:.4f}")
    report.append(f"Self Coupling (t)    : {t:.4f}")
    report.append(f"Loss Factor (a)      : {a:.4f}")
    report.append("")

    report.append("RESONANCE INFORMATION")
    report.append("-" * 60)
    report.append(f"Detected Resonances : {len(resonances)}")
    report.append("")

    for i, wavelength in enumerate(resonances, start=1):
        report.append(
            f"{i:02d}. {wavelength * 1e9:.6f} nm"
        )

    report.append("")
    report.append("AVERAGE PERFORMANCE METRICS")
    report.append("-" * 60)

    report.append(
        f"Average FSR          : {average_fsr * 1e9:.6f} nm"
    )

    report.append(
        f"Average FWHM         : {average_fwhm * 1e9:.6f} nm"
    )

    report.append(
        f"Average Loaded Q     : {average_q:.2f}"
    )

    report.append(
        f"Average ER           : {average_er:.2f} dB"
    )

    report.append(
        f"Insertion Loss       : {insertion_loss:.4f} dB"
    )

    report.append(
        f"Average Finesse      : {average_finesse:.2f}"
    )

    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)

    save_text_report(
        results_path,
        report_text
    )

    # ==========================================================
    # Build CSV Report
    # ==========================================================

    header = [
        "Radius_um",
        "Effective_Index",
        "Self_Coupling_t",
        "Loss_Factor_a",
        "Number_of_Resonances",
        "Average_FSR_nm",
        "Average_FWHM_nm",
        "Average_Q",
        "Average_ER_dB",
        "Insertion_Loss_dB",
        "Average_Finesse"
    ]

    rows = [[
        radius * 1e6,
        n_eff,
        t,
        a,
        len(resonances),
        average_fsr * 1e9,
        average_fwhm * 1e9,
        average_q,
        average_er,
        insertion_loss,
        average_finesse
    ]]

    save_csv_report(
        results_path,
        header,
        rows
    )

    print("\nCharacterization reports saved successfully.")

    print(
        f"Text Report : "
        f"{results_path / 'characterization_report.txt'}"
    )

    print(
        f"CSV Report  : "
        f"{results_path / 'characterization_report.csv'}"
    )