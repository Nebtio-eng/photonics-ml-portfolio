import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from src.physics.phase_model import phase_difference
from src.physics.mzi_transfer import (
    mzi_transmission
)

from src.robustness.robust_analysis import (
    robust_transmission
)


def evaluate_design():

    try:

        L1 = float(entry_L1.get()) * 1e-6
        L2 = float(entry_L2.get()) * 1e-6

        wavelength = (
            float(entry_wavelength.get()) * 1e-9
        )

        neff = float(entry_neff.get())

        delta_phi = phase_difference(
            neff,
            L1,
            neff,
            L2,
            wavelength
        )

        transmission = mzi_transmission(
            delta_phi
        )

        robust = robust_transmission(
            neff,
            L1,
            L2,
            wavelength
        )

        mean_T = robust["mean_transmission"]
        std_T = robust["std_transmission"]

        if mean_T > 0.8 and std_T < 0.05:
            verdict = "PASS"
            verdict_color = "green"

        else:
            verdict = "FAIL"
            verdict_color = "red"

        result_box.config(
            state="normal"
        )

        result_box.delete(
            "1.0",
            tk.END
        )

        result_box.insert(
            tk.END,
            f"""
========== MZI REPORT ==========

Ideal Transmission:
{transmission:.4f}

Robust Mean Transmission:
{mean_T:.4f}

Transmission Std Dev:
{std_T:.4f}

Verdict:
{verdict}

================================
"""
        )

        result_box.config(
            state="disabled"
        )

        verdict_label.config(
            text=verdict,
            foreground=verdict_color
        )

    except Exception as e:

        messagebox.showerror(
            "Input Error",
            str(e)
        )


root = tk.Tk()

root.title(
    "Mach-Zehnder Design Assistant"
)

root.geometry("600x500")

# Title

title = ttk.Label(
    root,
    text="MZI Design Assistant",
    font=("Arial", 18)
)

title.pack(pady=10)

# Frame

frame = ttk.Frame(root)
frame.pack(pady=10)

# Inputs

ttk.Label(
    frame,
    text="L1 (µm)"
).grid(row=0, column=0)

entry_L1 = ttk.Entry(frame)
entry_L1.grid(row=0, column=1)

ttk.Label(
    frame,
    text="L2 (µm)"
).grid(row=1, column=0)

entry_L2 = ttk.Entry(frame)
entry_L2.grid(row=1, column=1)

ttk.Label(
    frame,
    text="Wavelength (nm)"
).grid(row=2, column=0)

entry_wavelength = ttk.Entry(frame)
entry_wavelength.grid(row=2, column=1)

ttk.Label(
    frame,
    text="Effective Index"
).grid(row=3, column=0)

entry_neff = ttk.Entry(frame)
entry_neff.grid(row=3, column=1)

# Button

evaluate_button = ttk.Button(
    root,
    text="Evaluate Design",
    command=evaluate_design
)

evaluate_button.pack(pady=10)

# Verdict Label

verdict_label = ttk.Label(
    root,
    text="",
    font=("Arial", 16)
)

verdict_label.pack()

# Result Box

result_box = tk.Text(
    root,
    height=15,
    width=60
)

result_box.pack(pady=10)

result_box.config(state="disabled")

root.mainloop()