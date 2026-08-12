"""
Inference Script — Causal Autoregressive Tracking
TrackRAD2025 Grand Challenge — EEN1095 Implementation Project

Author: Chetan Kumar (A00054853)
Dublin City University, August 2026

This script loads the saved checkpoint and runs causal autoregressive
tracking on a specified patient sequence. Use this to reproduce the
published results without retraining.

Published results (Patient A_001, 100 frames):
    U-Net only:   Mean Dice = 0.8706
    VoxelMorph:   Mean Dice = 0.8686
    Combined:     Mean Dice = 0.8705

Usage:
    python inference.py

Configuration:
    CHECKPOINT_PATH — path to unet_conditioned_best.pth
    DATA_ROOT       — path to trackrad2025_labeled_training_data_50/
    PATIENT         — patient ID to track (e.g. 'A_001')
    N_PREV          — must match the checkpoint (default: 1)
    SAVE_VIDEO      — whether to save tracking video as MP4
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.models.unet import UNet
from src.utils.losses import dice_score, com_distance, dosimetric_acc

# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "/content/drive/MyDrive/TrackRad2025/checkpoints/unet_conditioned_best.pth"
DATA_ROOT       = "/content/drive/MyDrive/TrackRad2025/trackrad2025_labeled_training_data_50"
PATIENT         = "A_001"
N_PREV          = 1
TARGET_SIZE     = 256
SAVE_VIDEO      = True
VIDEO_PATH      = f"/content/drive/MyDrive/TrackRad2025/results/{PATIENT}_tracking.mp4"
ALPHA           = 0.7    # fusion weight (U-Net contribution)
MAX_GROWTH      = 1.15   # maximum mask area growth per frame

# ── Setup ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model = UNet(in_channels=1 + N_PREV, out_channels=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"Checkpoint loaded: {CHECKPOINT_PATH}")

# Load patient data
patient_path = os.path.join(DATA_ROOT, PATIENT)
frames_path  = os.path.join(patient_path, "images",  f"{PATIENT}_frames.mha")
labels_path  = os.path.join(patient_path, "targets", f"{PATIENT}_labels.mha")

frames = sitk.GetArrayFromImage(sitk.ReadImage(frames_path))
labels = sitk.GetArrayFromImage(sitk.ReadImage(labels_path))

# (H, W, T) -> (T, H, W)
frames = np.transpose(frames, (2, 0, 1)).astype(np.float32)
labels = np.transpose(labels, (2, 0, 1)).astype(np.float32)
frames = frames / (frames.max() + 1e-8)

T = frames.shape[0]
print(f"Patient {PATIENT}: {T} frames, shape {frames.shape}")

# ── Resize helpers ────────────────────────────────────────────────────────────

def resize_frame(arr):
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (TARGET_SIZE, TARGET_SIZE),
                         mode="bilinear", align_corners=False).squeeze()

def resize_mask(arr):
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (TARGET_SIZE, TARGET_SIZE),
                         mode="nearest").squeeze()

# ── Causal autoregressive tracking ───────────────────────────────────────────

print("\nRunning causal tracking...")
predicted_masks = []
dice_scores     = []
com_scores      = []
dos_scores      = []

# First frame mask provided by challenge interface
prev_mask = resize_mask(labels[0])
prev_size = int(prev_mask.sum().item())

with torch.no_grad():
    for t in range(T):
        frame_t = resize_frame(frames[t])
        gt      = resize_mask(labels[t])

        # Stack frame + N previous masks
        x = torch.stack([frame_t, prev_mask], dim=0).unsqueeze(0).to(device)

        # U-Net prediction
        pred     = torch.sigmoid(model(x)).squeeze().cpu()
        pred_bin = (pred > 0.5).float()

        # Mask growth constraint — prevent runaway expansion
        curr_size = int(pred_bin.sum().item())
        if prev_size > 0 and curr_size > prev_size * MAX_GROWTH:
            for thresh in [0.55, 0.60, 0.65, 0.70]:
                pred_bin = (pred > thresh).float()
                if int(pred_bin.sum().item()) <= prev_size * MAX_GROWTH:
                    break

        predicted_masks.append(pred_bin.numpy())

        # Compute per-frame metrics
        gt_np = gt.numpy().astype(np.uint8)
        p_np  = pred_bin.numpy().astype(np.uint8)

        inter = (pred_bin * gt).sum()
        d     = (2 * inter + 1e-6) / (pred_bin.sum() + gt.sum() + 1e-6)
        dice_scores.append(d.item())
        com_scores.append(com_distance(p_np, gt_np))
        dos_scores.append(dosimetric_acc(p_np, gt_np))

        prev_mask = pred_bin
        prev_size = int(pred_bin.sum().item())

        if t % 20 == 0:
            print(f"  Frame {t+1:3d}/{T} | "
                  f"Dice: {dice_scores[-1]:.4f} | "
                  f"CoM: {com_scores[-1]:.2f}px | "
                  f"Dos: {dos_scores[-1]:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────

def safe_mean(lst):
    vals = [v for v in lst if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")

print(f"\n{'='*50}")
print(f"  TRACKING RESULTS — Patient {PATIENT}")
print(f"{'='*50}")
print(f"  Mean Dice:          {safe_mean(dice_scores):.4f}")
print(f"  Mean CoM Distance:  {safe_mean(com_scores):.2f} px")
print(f"  Mean Dos. Accuracy: {safe_mean(dos_scores):.4f}")
print(f"  Min Dice:           {min(dice_scores):.4f}")
print(f"  Max Dice:           {max(dice_scores):.4f}")
print(f"{'='*50}")

# ── Dice curve plot ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
ax.plot(dice_scores, color="#2ECC71", linewidth=1.8, label="Per-frame Dice")
ax.axhline(y=safe_mean(dice_scores), color="#E67E22", linestyle="--",
           label=f"Mean: {safe_mean(dice_scores):.4f}")
ax.axhline(y=0.80, color="#E74C3C", linestyle="--",
           alpha=0.7, label="Target (0.80)")
ax.set_xlabel("Frame number"); ax.set_ylabel("Dice similarity coefficient")
ax.set_title(f"Causal Tracking — Patient {PATIENT}", fontweight="bold")
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"/content/drive/MyDrive/TrackRad2025/results/{PATIENT}_dice_curve.png",
            dpi=200, bbox_inches="tight")
plt.show()

# ── Tracking video ────────────────────────────────────────────────────────────

if SAVE_VIDEO:
    print("\nGenerating tracking video...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.axis("off"); ax.set_facecolor("black")
    axes[0].set_title("MRI Frame",      color="white", fontsize=12, fontweight="bold")
    axes[1].set_title("Ground Truth",   color="white", fontsize=12, fontweight="bold")
    axes[2].set_title("U-Net Tracking", color="white", fontsize=12, fontweight="bold")

    f0 = resize_frame(frames[0]).numpy()
    im0 = axes[0].imshow(f0, cmap="gray", vmin=0, vmax=1)
    im1 = axes[1].imshow(f0, cmap="gray", vmin=0, vmax=1)
    im2 = axes[2].imshow(f0, cmap="gray", vmin=0, vmax=1)
    dtxt = fig.text(0.5, 0.01,
                    f"Frame 1/{T} | Dice: {dice_scores[0]:.4f}",
                    ha="center", color="white", fontsize=11, fontweight="bold")

    def update(t):
        ft = resize_frame(frames[t]).numpy()
        gt = resize_mask(labels[t]).numpy()
        pt = predicted_masks[t]
        im0.set_array(ft); im1.set_array(ft); im2.set_array(ft)
        for coll in axes[1].collections: coll.remove()
        for coll in axes[2].collections: coll.remove()
        if gt.sum() > 0:
            axes[1].contour(gt, levels=[0.5], colors=["#2ECC71"], linewidths=2.5)
        if pt.sum() > 0:
            axes[2].contour(pt, levels=[0.5], colors=["#E74C3C"], linewidths=2.5)
        dtxt.set_text(f"Frame {t+1}/{T} | Dice: {dice_scores[t]:.4f} | "
                      f"Mean: {safe_mean(dice_scores):.4f}")
        return [im0, im1, im2]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=80, blit=False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    ani.save(VIDEO_PATH, writer=animation.FFMpegWriter(fps=10, bitrate=1800), dpi=150)
    print(f"Video saved: {VIDEO_PATH}")
