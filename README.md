# EEN1095 Project Repository

## Real-Time Lung Tumour Tracking in Cine-MRI Using Deep Learning

**Student:** Chetan Kumar | A00054853
**Programme:** MEng Electronics and Computer Engineering
**Supervisor:** Prof. Robert Sadleir
**Module:** EEN1095 Implementation Project
**Institution:** Dublin City University
**Date:** August 2026

---

## Project Overview

This repository implements a real-time lung tumour tracking pipeline for the TrackRAD2025 Grand Challenge. The system combines a mask-conditioned U-Net segmentation module with a VoxelMorph deformable image registration module, fused into a causal tracking pipeline.

The central finding of this project is that the TrackRAD2025 tracking task is fundamentally a **mask propagation problem** rather than a frame-independent segmentation problem. Conditioning the U-Net on the previous frame's tumour mask (supplied by the challenge interface) raises the Dice similarity coefficient from 0.3776 (blind segmentation) to 0.9426 (guided propagation), matching state-of-the-art performance with 7.77 million parameters.

---

## Key Results

| Method | Dice | CoM Distance | Dosimetric Acc |
|--------|------|-------------|----------------|
| First-frame copy (no AI) | ~0.70 | — | — |
| Blind U-Net (1 channel) | 0.3776 | — | — |
| DINOMotion (SOTA, ~330M params) | ~0.92 | — | — |
| **Mask-conditioned U-Net (N=1)** | **0.9409** | **0.67 px** | **0.9997** |
| Mask-conditioned U-Net (N=3) | 0.9417 | 0.64 px | 0.9999 |
| Mask-conditioned U-Net (N=5) | 0.9426 | 0.64 px | 0.9999 |

**Causal autoregressive tracking (Patient A_001, 100 frames):**

| Configuration | Mean Dice |
|---|---|
| U-Net only | 0.8706 |
| VoxelMorph only | 0.8686 |
| Combined (alpha=0.7) | 0.8705 |

---

## Repository Structure

```
EEN1095ProjectRepo/
|-- README.md
|-- requirements.txt
|-- train.py              Main U-Net training script
|-- inference.py          Causal tracking using saved checkpoint
|-- src/
|   |-- models/
|   |   |-- unet.py       Mask-conditioned U-Net (7.77M params)
|   |-- data/
|   |   |-- dataset.py    TrackRAD dataset loader + lazy loading version
|   |-- utils/
|       |-- losses.py     Loss functions + all 4 official metrics
|-- results/
|   |-- ablation_study_final.png
|   |-- tracking_qualitative.png
|   |-- pipeline_comparison.png
|   |-- training_curves.png
|   |-- voxelmorph_results.png
|   |-- A_001_tracking.mp4
|-- checkpoints/
    |-- unet_conditioned_best.pth   Best U-Net (Dice 0.9409)
    |-- voxelmorph_best.pth         Best VoxelMorph (NCC -0.4619)
```

---

## Reproducing Published Results

**To reproduce results using the saved checkpoint (recommended):**

```bash
git clone https://github.com/chetanx88/EEN1095ProjectRepo.git
cd EEN1095ProjectRepo
pip install -r requirements.txt
python inference.py
```

Update `CHECKPOINT_PATH` and `DATA_ROOT` in `inference.py` to point to your local paths before running.

**Note:** The provided checkpoint `unet_conditioned_best.pth` is the exact model used to generate all results in the portfolio. Loading this checkpoint and running inference on the same data will reproduce the published Dice scores exactly.

**To retrain from scratch:**

```bash
python train.py
```

Update `ROOT` and `SAVE_DIR` in `train.py` before running. Set `N_PREV = 1` for the main result, or `3`/`5` for the ablation. Note that retraining will produce similar but not bit-identical results due to stochastic weight initialisation and data augmentation.

---

## Dataset

TrackRAD2025 — 50 labeled training patients across 3 centres (A: 0.35T, B/C: 1.5T).

Download from HuggingFace:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LMUK-RADONC-PHYS-RES/TrackRAD2025",
    repo_type="dataset",
    local_dir="/path/to/trackrad2025",
    allow_patterns=["trackrad2025_labeled_training_data/*"]
)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.12, PyTorch 2.0+, SimpleITK 2.2+, NumPy, SciPy, Matplotlib

---

## References

1. Ronneberger et al. (2015) — U-Net
2. Balakrishnan et al. (2019) — VoxelMorph
3. Salari et al. (2024) — DINOMotion
4. Lombardo et al. (2025) — TrackRAD2025 dataset

## Model Checkpoints

| File | Link |
|---|---|
| `unet_conditioned_best.pth` (Best U-Net, Dice 0.9409) | [Download]([YOUR_DRIVE_LINK](https://drive.google.com/file/d/1hhw7kG_Y81lCFfUlLevbCJ8Gj4kISsgB/view?usp=sharing)) |
| `voxelmorph_best.pth` (Best VoxelMorph, NCC -0.4619) | [Download]([YOUR_DRIVE_LINK](https://drive.google.com/file/d/1FF0yJKt36jJb959EbeOTNV7e0xLb8Ilf/view?usp=sharing)) |
