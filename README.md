# An-identifiable-denoisy-autoencoder-aligning-clinical-and-genomic-data
A PyTorch-based denoising autoencoder inspired by ICE-BeeM's identifiable architecture. Adapted MLP model for aligning clinical-genomic data, emphasizing identifiability to accurately invert genomic variations from clinical changes.

## Overview
This repository contains a PyTorch implementation of an identifiable denoising autoencoder designed to align high-dimensional clinical and genomic data. The model is autonomously constructed based on a Multi-Layer Perceptron (MLP) foundation, drawing inspiration from the ICE-BeeM  framework presented in the NeurIPS 2020 paper. 

Key adaptations ensure compliance with ICE-BeeM's requirements for model architecture while tailoring it for bioinformatics applications. The primary goal is to model coordinated variations between clinical variables (e.g., patient phenotypes) and genomic data (e.g., expression profiles), enabling accurate inversion: by perturbing clinical inputs, the model reconstructs corresponding genomic changes with high fidelity, leveraging identifiability to produce biologically meaningful representations.  

This project demonstrates strong engineering capabilities in deep learning, with a focus on interpretability over black-box modeling. It is ongoing, with code available for refinement and extension.

## Features

- **ICE-BeeM Compliance and Adaptation**: The architecture satisfies the foundational requirements outlined in the ICE-BeeM paper for identifiability in conditional energy-based models. Modifications include non-monotonic augmentations (e.g., z, z², 0.1z³) in encoders to enhance representation uniqueness, while maintaining MLP simplicity for rapid adaptation.
  
- **Identifiability for Inversion**: Emphasizes mathematical optimization for interpretable representations. When clinical data is altered, the model accurately inverts and reconstructs genomic patterns, maximizing biological relevance and minimizing ambiguity in high-dimensional alignments.

- **Modular Design**: 
  - **Encoder**: Handles genomic input with optional non-monotonic transformations for enhanced expressivity.
  - **ConditionEncoder**: Processes clinical conditioning variables, mirroring the encoder's augmentation for consistency.
  - **Decoder**: Reconstructs outputs with noise injection for robustness, using Kaiming initialization.

- **Frontier DL Integration**: Incorporates spectral normalization and golden ratio-based hidden dimensions for stable training and theoretical alignment with advanced deep learning principles.

- **Applications**: Ideal for clinical bioinformatics tasks, such as predicting genomic responses to clinical interventions or integrating multi-omics data.

## Installation

1. Clone the repository:
