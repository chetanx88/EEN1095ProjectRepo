# EEN1095 Project Repository
## Real-Time Lung Tumor Tracking in Cine-MRI Using Deep Learning

**Student:** Chetan Kumar | A00054853
**Programme:** MEng Electronics and Computer Engineering
**Supervisor:** Prof. Robert Sadleir
**Module:** EEN1095 Implementation Project
**Institution:** Dublin City University

## Project Overview
Real-time lung tumor tracking pipeline for the TrackRAD2025 Grand Challenge.
Mask-conditioned U-Net: current MRI frame + previous tumor mask as 2-channel input.
This framing converts the task from blind segmentation to guided mask propagation.

## Key Results

| Method | Dice | CoM Distance | Dosimetric Acc |
|--------|------|-------------|----------------|
| Copy-paste baseline (no AI) | ~0.70 | - | - |
| Blind U-Net (1-channel) | 0.37 | - | - |
| Mask-conditioned U-Net (N=1) | **0.9430** | 0.67px | 0.9997 |
| DINOMotion (state-of-the-art) | ~0.92 | - | - |

## How It Works
Standard U-Net: 1 input channel (MRI frame) -> Dice 0.37
Mask-conditioned U-Net: 2 input channels (MRI frame + previous mask) -> Dice 0.9430

## Repository Structure
    EEN1095ProjectRepo/
    |-- src/
    |   |-- models/unet.py       # Mask-conditioned U-Net
    |   |-- data/dataset.py      # TrackRAD dataset loader
    |   |-- utils/losses.py      # Loss functions + 4 eval metrics
    |-- train.py                 # Main training script
    |-- requirements.txt
    |-- README.md

## Dataset
TrackRAD2025 - 50 labeled patients, 3 centres (A: 0.35T, B/C: 1.5T)
HuggingFace: https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025

## Installation
    git clone https://github.com/chetanx88/EEN1095ProjectRepo.git
    cd EEN1095ProjectRepo
    pip install -r requirements.txt

## Training
    python train.py

## References
1. Ronneberger et al. (2015) - U-Net
2. Balakrishnan et al. (2019) - VoxelMorph
3. Salari et al. (2024) - DINOMotion
4. Oktay et al. (2018) - Attention U-Net
5. Isensee et al. (2021) - nnU-Net
