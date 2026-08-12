"""
Training Script — Mask-Conditioned U-Net
TrackRAD2025 Grand Challenge — EEN1095 Implementation Project

Author: Chetan Kumar (A00054853)
Dublin City University, August 2026

Usage:
    python train.py

Key configuration variables (edit before running):
    ROOT        — path to trackrad2025_labeled_training_data_50/
    SAVE_DIR    — directory to save model checkpoints
    N_PREV      — number of previous masks to stack (1, 3, or 5)
    NUM_EPOCHS  — training epochs (60 for main run, 30 for ablation)

Reproducibility note:
    The train/val split uses random.seed(42). Training itself involves
    stochastic weight initialisation, augmentation, and batch ordering.
    To reproduce the published results exactly, load the provided
    checkpoint (unet_conditioned_best.pth) rather than retraining.
    See inference.py for checkpoint-based evaluation.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.data.dataset import TrackRADLazyDataset
from src.utils.losses import combined_loss, dice_score, evaluate_batch

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT       = "/content/drive/MyDrive/TrackRad2025/trackrad2025_labeled_training_data_50"
SAVE_DIR   = "/content/drive/MyDrive/TrackRad2025/checkpoints"
N_PREV     = 1         # 1, 3, or 5
NUM_EPOCHS = 60        # 60 for main run; 30 for ablation
BATCH_SIZE = 16
LR         = 3e-4
WEIGHT_DECAY = 1e-5

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Train / validation split ──────────────────────────────────────────────────

all_patients = sorted([p for p in os.listdir(ROOT) if not p.startswith(".")])

# Stratified 80/20 split — preserves centre distribution in both sets
by_centre = defaultdict(list)
for p in all_patients:
    by_centre[p[0]].append(p)

train_ids, val_ids = [], []
for centre, pts in sorted(by_centre.items()):
    random.seed(42)
    random.shuffle(pts)
    split = int(0.8 * len(pts))
    train_ids += pts[:split]
    val_ids   += pts[split:]

print(f"Total patients: {len(all_patients)}")
print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

# Centre breakdown
for name, ids in [("Train", train_ids), ("Val", val_ids)]:
    counts = defaultdict(int)
    for p in ids:
        counts[p[0]] += 1
    print(f"  {name} centres: {dict(sorted(counts.items()))}")

# ── Data loaders ──────────────────────────────────────────────────────────────

train_ds = TrackRADLazyDataset(ROOT, train_ids, augment=True,  n_prev_masks=N_PREV)
val_ds   = TrackRADLazyDataset(ROOT, val_ids,   augment=False, n_prev_masks=N_PREV)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

# ── Model, optimiser, scheduler ───────────────────────────────────────────────

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model     = UNet(in_channels=1 + N_PREV, out_channels=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Train frames: {len(train_ds)} | Val frames: {len(val_ds)}")
print(f"N_PREV={N_PREV} | Epochs={NUM_EPOCHS} | Batch={BATCH_SIZE} | LR={LR}\n")

# ── Training loop ─────────────────────────────────────────────────────────────

best_dice = 0.0
history   = {"train_loss": [], "val_dice": [], "val_sd95": [],
             "val_com": [], "val_dos": []}

for epoch in range(NUM_EPOCHS):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for x, masks in train_loader:
        x, masks = x.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = combined_loss(model(x), masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    all_metrics = defaultdict(list)
    with torch.no_grad():
        for x, masks in val_loader:
            x, masks = x.to(device), masks.to(device)
            pred = model(x)
            m    = evaluate_batch(pred, masks)
            for k, v in m.items():
                if not np.isnan(v):
                    all_metrics[k].append(v)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    epoch_metrics = {k: safe_mean(v) for k, v in all_metrics.items()}
    scheduler.step()

    # ── Save best ─────────────────────────────────────────────────────────────
    if epoch_metrics["dice"] > best_dice:
        best_dice = epoch_metrics["dice"]
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, f"unet_stack{N_PREV}_best.pth")
        )
        saved = "  <- best saved"
    else:
        saved = ""

    for k, v in epoch_metrics.items():
        history[f"val_{k}"].append(v)
    history["train_loss"].append(train_loss)

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Dice: {epoch_metrics['dice']:.4f} | "
          f"SD95: {epoch_metrics['sd95']:.2f}px | "
          f"CoM: {epoch_metrics['com']:.2f}px | "
          f"Dos: {epoch_metrics['dos']:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}"
          f"{saved}")

print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
print(f"Checkpoint saved to: {SAVE_DIR}/unet_stack{N_PREV}_best.pth")
