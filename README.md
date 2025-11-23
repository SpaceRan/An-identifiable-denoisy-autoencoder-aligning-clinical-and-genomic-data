# An-identifiable-denoisy-autoencoder-aligning-clinical-and-genomic-data

A PyTorch-based denoising autoencoder inspired by ICE-BeeM's identifiable architecture. Adapted MLP model for aligning clinical-genomic data, emphasizing identifiability to accurately invert genomic variations from clinical changes.

## Overview

This repository contains a PyTorch implementation of an identifiable denoising autoencoder designed to align high-dimensional clinical and genomic data. The model is autonomously constructed based on a Multi-Layer Perceptron (MLP) foundation, drawing inspiration from the ICE-BeeM framework presented in the NeurIPS 2020 paper.

Key adaptations ensure compliance with ICE-BeeM's requirements for model architecture while tailoring it for bioinformatics applications. The primary goal is to model coordinated variations between clinical variables (e.g., patient phenotypes) and genomic data (e.g., expression profiles), enabling accurate inversion: by perturbing clinical inputs, the model reconstructs corresponding genomic changes with high fidelity, leveraging identifiability to produce biologically meaningful representations.

## Features

- **ICE-BeeM Compliance and Adaptation**: The architecture draws from ICE-BeeM's identifiability principles in conditional energy-based models. Includes LeakyReLU activations, optional non-monotonic augmentations (e.g., z, z², 0.1z³) in encoders for better representation uniqueness, and spectral normalization (SN) to ensure Lipschitz continuity and training stability—all while keeping MLP simplicity for quick tweaks.

- **Identifiability for Inversion**: Focuses on mathematical tweaks for interpretable outputs. Altering clinical data lets the model reconstruct genomic patterns accurately, boosting biological relevance in high-dim alignments.

- **Modular Design**:
  - **Encoder**: Processes genomic input with optional non-monotonic transforms for more expressivity.
  - **ConditionEncoder**: Handles clinical conditions, matching the encoder's augmentations for alignment.
  - **Decoder**: Reconstructs with light noise injection for robustness, plus Kaiming init.

The code is in `src/`. This is an ongoing project—feel free to check the modules and suggest refinements.
