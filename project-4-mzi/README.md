# Project 4 — Mach–Zehnder Interferometer Design Platform

## Overview

This project implements a physics-guided design workflow for a silicon photonic Mach–Zehnder Interferometer (MZI).

The platform combines analytical photonic modeling, machine learning surrogates, robustness evaluation, optimization, and GUI-based design verification.

---

# Objectives

* Model optical phase accumulation
* Simulate interference-based transmission
* Generate physics-consistent datasets
* Train a machine learning surrogate
* Evaluate robustness under parameter variation
* Explore design tradeoffs using Pareto analysis
* Build a GUI-assisted design checker

---

# Mach–Zehnder Physics

The phase difference between the two interferometer arms is

Δφ = (2π n_eff ΔL) / λ

where:

* Δφ = phase difference
* n_eff = effective index
* ΔL = arm length difference
* λ = operating wavelength

The resulting transmission is

T = cos²(Δφ / 2)

This relationship forms the foundation of all simulations in this project.

---

# Workflow

Physics Model

↓

Dataset Generation

↓

ML Surrogate Training

↓

Robustness Analysis

↓

Pareto Optimization

↓

GUI Design Verification

---

# Repository Structure

```text
project4_mzi/

├── models/
│   └── mzi_surrogate.joblib
│
├── results/
│   ├── mzi_dataset.csv
│   ├── transmission_vs_deltaL.png
│   └── pareto_front.png
│
├── src/
│   ├── physics/
│   ├── data/
│   ├── ml/
│   ├── robustness/
│   ├── analysis/
│   └── gui/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Features

### Physics Layer

* Analytical phase model
* MZI transmission model
* Physics-consistent parameter sweeps

### Machine Learning Layer

* Surrogate regression model
* Fast transmission prediction
* Saved trained model

### Robustness Layer

* Parameter variation analysis
* Design sensitivity evaluation
* Decision-time robustness checks

### Optimization Layer

* Pareto front generation
* Tradeoff visualization
* Design exploration

### GUI Layer

* User-defined design parameters
* Transmission estimation
* PASS / FAIL design assessment

---

# Running the Project

Generate Dataset

```bash
python -m src.data.generate_dataset
```

Train ML Surrogate

```bash
python -m src.ml.train_model
```

Load Saved Model

```bash
python -m src.ml.load_model
```

Run Robustness Analysis

```bash
python -m src.robustness.robust_analysis
```

Generate Final Figures

```bash
python -m src.analysis.final_plots
```

Launch GUI

```bash
python -m src.gui.mzi_gui
```

---

# Results

Generated artifacts include:

* Physics-generated MZI dataset
* Transmission response curves
* Pareto optimization plots
* Trained ML surrogate model

---

# Engineering Lessons

This project demonstrates:

* Physics-guided machine learning
* Separation of physics and robustness layers
* Surrogate modeling for photonic devices
* Reproducible engineering workflows
* Design-oriented photonics software development

---

# Author

Photonics + Machine Learning Design Portfolio

Project 4 of the integrated photonics engineering roadmap.
