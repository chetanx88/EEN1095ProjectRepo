
import os, random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader

from src.data.dataset import TrackRADDataset
from src.models.unet import UNet
from src.utils.losses import combined_loss, dice_score

ROOT       = "/content/drive/MyDrive/TrackRad2025/trackrad2025_labeled_training_data_50"
SAVE_DIR   = "/content/drive/MyDrive/TrackRad2025/outputs"
N_PREV     = 1          # number of previous masks to stack
NUM_EPOCHS = 40
BATCH_SIZE = 8
LR         = 3e-4

os.makedirs(SAVE_DIR, exist_ok=True)

all_patients = sorted([p for p in os.listdir(ROOT) if not p.startswith(".")])
by_centre    = defaultdict(list)
for p in all_patients:
    by_centre[p[0]].append(p)

train_ids, val_ids = [], []
for centre, pts in sorted(by_centre.items()):
    random.seed(42); random.shuffle(pts)
    split = int(0.8 * len(pts))
    train_ids += pts[:split]; val_ids += pts[split:]

train_loader = DataLoader(TrackRADDataset(ROOT, train_ids, augment=True,  n_prev_masks=N_PREV),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(TrackRADDataset(ROOT, val_ids,   augment=False, n_prev_masks=N_PREV),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = UNet(in_channels=1+N_PREV, out_channels=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

best_dice = 0.0
for epoch in range(NUM_EPOCHS):
    model.train(); train_loss = 0
    for x, masks in train_loader:
        x, masks = x.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = combined_loss(model(x), masks)
        loss.backward(); optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval(); val_dice = 0
    with torch.no_grad():
        for x, masks in val_loader:
            x, masks = x.to(device), masks.to(device)
            val_dice += dice_score(model(x), masks)
    val_dice /= len(val_loader)
    scheduler.step()

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"unet_stack{N_PREV}_best.pth"))
        saved = "  <- best saved"
    else:
        saved = ""

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | "
          f"Val Dice: {val_dice:.4f}{saved}")

print(f"Done! Best Val Dice: {best_dice:.4f}")
