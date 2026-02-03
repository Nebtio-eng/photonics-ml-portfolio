# Directional Coupler Design Tool (Integrated Photonics)

## Overview
This project presents a physics-guided, ML-assisted design tool for silicon
photonic directional couplers. The goal is not just to design a 50:50 splitter,
but to identify geometries that remain balanced under fabrication and wavelength
variations.

The tool combines analytical coupled-mode theory, machine-learning surrogate
modeling, Pareto tradeoff analysis, and a GUI-based design checker.

## Key Concepts
- Evanescent coupling
- Periodic power transfer
- Gap vs length tradeoff
- Physics-guided ML surrogate
- Robust design under fabrication variation

## Workflow
1. Physics-based simulation of power transfer
2. Dataset generation over design space
3. ML regression trained on ideal physics
4. Robustness-aware design checking
5. Interactive GUI for geometry validation

## Design Philosophy
The ML model is trained on ideal (noise-free) physics to learn the true coupling
behavior. Fabrication variation, wavelength shift, and loss are injected at
decision time using Monte Carlo analysis, mimicking real photonic design flows.

## Results
- Multiple valid 50:50 designs exist
- Compact designs are more fabrication-sensitive
- Robust designs trade area for tolerance
- Pareto fronts reveal optimal compromises

## Tools Used
- Python, NumPy, Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Conda (WSL Ubuntu)

## Key Takeaway
Directional couplers do not have a single best geometry—only designs that balance
performance, robustness, and footprint.


### Reproducibility
Trained ML models are intentionally excluded from version control.
Run the following to regenerate them:

```bash
python src/train_regression.py
