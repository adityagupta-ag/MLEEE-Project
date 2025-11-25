# 2D Field Reconstruction from Sparse Sensor Data

This repository contains PyTorch implementations for reconstructing 2D physical fields (such as flow, temperature, etc.) from sparse sensor data using deep learning. The models are trained and tested on synthetic data to benchmark the effectiveness of deep convolutional and residual CNN architectures.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Deep Learning Models](#deep-learning-models)
- [Experiments](#experiments)
- [How to Run](#how-to-run)
- [Results](#results)
- [Files in this Repo](#files-in-this-repo)
---

## Overview

The goal is to reconstruct a 2D field (e.g., vorticity, temperature, etc.) from a limited set of noisy sensor measurements, using:
- Simple 7-layer CNN
- Deep ResNet-based CNN

Multiple setups are tested, varying the number of sensors, field snapshots, and network depth.

---

## Data

You will need the following files to run the experiments (included or generated/noted in `data_generation.ipynb`):
- `record_x.csv` — X-coordinates of the grid
- `record_y.csv` — Y-coordinates of the grid
- `ch_2Dxysec.pickle` — 4D array with simulated field snapshots (NEEDS TO BE DOWNLOADED FROM data_generation.ipynb file)
- (`Model_2Dxysec_pytorch.pth`, `Model_ResNet_2Dxysec_weights.pth` are model checkpoints generated after training)

---

## Deep Learning Models

**SimpleCNNReconstructor**

This model implements a deep feed-forward convolutional neural network inspired by traditional image super-resolution architectures:

- **Input:** 2 channels  
  (1) Nearest-neighbor interpolation of sparse field measurements,  
  (2) binary mask indicating sensor locations.
- **Architecture:** 
  - A 7-layer structure:
    - The first layer is a 7x7 convolution with 48 channels.
    - This is followed by a stack of six [7x7 convolution + ReLU] pairs, all with 48 channels.
    - A final 3x3 convolution brings the output back to a single channel.
- **Purpose:** Learns to upsample/interpolate sparse field observations to reconstruct the full dense field.

**ResNetReconstructor**

A deeper and more sophisticated architecture leveraging ideas from ResNet:

- **Input:** Same as above (nearest-neighbor field + mask).
- **Architecture:**
  - Starts with a 5x5 convolution and batch normalization for initial feature extraction.
  - Multiple *Residual Blocks* are used, each consisting of two convolutional layers, batch normalization, and spatial dropout. Each block allows the network to learn identity mappings, aiding gradient flow and deeper feature learning.
  - Block structure:
    - The network increases and then decreases feature map depth through the sequence:  
      [64→64, 64→64, 64→128, 128→128, (optional: extra 128→128 blocks), 128→64], ending in a 3x3 convolution to 1 channel.
- **Purpose:** The skip connections (residuals) make this model better suited for capturing complex features and gradients, yielding better reconstructions with the potential for deeper networks.

---

## Experiments

The following experiments are implemented in `test.ipynb`:
1. **CNN** — 50 sensors, 500 snapshots, 500 epochs
2. **ResNet** — 50 sensors, 500 snapshots, 200 epochs
3. **CNN** — 100 sensors, 5000 snapshots, 200 epochs
4. **ResNet** — 100 sensors, 500 snapshots, 200 epochs
5. **ResNet** — 100 sensors, 5000 snapshots, 200 epochs

Each experiment follows:
- Data prep (sparse measurement simulation, interpolation, train/test split)
- Model training with early stopping 
- Visualization of test predictions (field reconstruction comparison)
- Reports test MSE

---

## How to Run

1. **Install dependencies** (via pip or conda):
    - torch, numpy, pandas, matplotlib, tqdm, scikit-learn, scipy

2. **Ensure data files are present** in the repository root. 

**Note:** The pickle file needs to be downloaded from the link given in 'data_generation.ipynb'

3. **Run**:
    - `data_generation.ipynb` *(to generate or download data if needed)*
    - `test.ipynb` *(to reproduce experiments and results)*

**Note:** GPU acceleration with CUDA is supported and recommended for faster training.

---

## Results

Predictions are visualized and quantitative errors (MSE/RMSE) are logged for each configuration.

**Example of Field Reconstruction Result (at 200 Epochs):**

![Field Reconstruction Example (200 epochs)](Result_images/reconstruction_200E.png)

*From left to right: Ground Truth Field, Nearest Neighbor Interpolation, Model Reconstruction (after 200 training epochs).*

Model history and performance are saved as CSVs:
- `Model_2Dxysec_pytorch.csv`, `Model_2Dxysec_pytorch_100_sensors.csv` — Training results for various runs

---

## Files in this Repo

- `data_generation.ipynb` — Data creation/download and preprocessing
- `test.ipynb` — Main experiments and all model benchmarking logic
- `record_x.csv`, `record_y.csv` — Grid coordinates
- `ch_2Dxysec.pickle` — Synthetic field dataset
- `Model_2Dxysec_pytorch.pth`, `Model_ResNet_2Dxysec_weights.pth` — Model checkpoint (weights)
- `Model_2Dxysec_pytorch.csv` — Training/validation logs

---