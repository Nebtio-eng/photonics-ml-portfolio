# Lessons Learned

## What Worked

- Analytical simulator
- Dataset generation
- Modular software architecture
- Training pipeline
- Evaluation workflow

## What Did Not Work

The neural network struggled to reproduce narrow resonance dips.

Even after:

- Output normalization
- Batch Normalization
- AdamW
- Learning-rate scheduling

the model still produced overly smooth spectra.

## Root Cause

The learning problem itself is difficult because the model maps:

4 parameters

↓

1024 outputs

using MSE loss.

The optimizer was not the bottleneck.

## Future Research

A more suitable formulation would predict:

Radius
Effective Index
Coupling
Loss
Wavelength

↓

Transmission

This converts the task from

4 → 1024

to

5 → 1

which is generally easier to learn and better captures localized spectral features.