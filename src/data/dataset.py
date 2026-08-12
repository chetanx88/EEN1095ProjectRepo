"""
TrackRAD2025 Dataset Loader
EEN1095 Implementation Project

Author: Chetan Kumar (A00054853)
Dublin City University, August 2026

Two dataset classes:
    TrackRADDataset     — pre-loads all frames into memory (float16)
                          suitable for N=1, may run out of RAM for N>3
    TrackRADLazyDataset — loads frames on demand from disk
                          RAM-safe for all N values, slower per epoch

Both classes implement:
    - MetaImage (.mha) reading via SimpleITK
    - Axis transposition (H,W,T) -> (T,H,W)
    - Per-sequence intensity normalisation to [0,1]
    - Spatial resampling to TARGET_SIZE x TARGET_SIZE
    - Mask conditioning: frame + N previous masks as input channels
    - Scheduled sampling corruption (training only)
    - Data augmentation: flips, rotation, brightness jitter
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import SimpleITK as sitk

TARGET_SIZE = 256  # spatial resolution for all inputs


# ── File discovery ────────────────────────────────────────────────────────────

def find_patient_files(root_dir: str, folder_name: str):
    """
    Locate frames and labels .mha files for a given patient folder.
    Handles both standard naming (A_001) and Google Drive duplicate
    naming (A_001 (1)) by scanning the images/ and targets/ directories.

    Returns:
        (frames_path, labels_path) or (None, None) if not found.
    """
    patient_path = os.path.join(root_dir, folder_name)
    if not os.path.exists(patient_path):
        return None, None

    images_dir  = os.path.join(patient_path, "images")
    targets_dir = os.path.join(patient_path, "targets")

    if not (os.path.exists(images_dir) and os.path.exists(targets_dir)):
        return None, None

    frame_files = [f for f in os.listdir(images_dir)
                   if f.endswith("_frames.mha")]
    label_files = [f for f in os.listdir(targets_dir)
                   if f.endswith("_labels.mha")]

    if frame_files and label_files:
        return (os.path.join(images_dir,  frame_files[0]),
                os.path.join(targets_dir, label_files[0]))

    return None, None


# ── Resize helpers ────────────────────────────────────────────────────────────

def _resize_frame(arr: np.ndarray, size: int = TARGET_SIZE) -> torch.Tensor:
    """Bilinear resize of a 2D frame array to (1, size, size)."""
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)   # (1, H, W)


def _resize_mask(arr: np.ndarray, size: int = TARGET_SIZE) -> torch.Tensor:
    """Nearest-neighbour resize of a 2D mask array to (1, size, size)."""
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(
        t, size=(size, size), mode="nearest"
    ).squeeze(0)   # (1, H, W)


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment(frame: torch.Tensor,
             prev_masks: list,
             mask: torch.Tensor):
    """
    Apply identical spatial augmentation to frame and all masks.
    Brightness jitter is applied to the frame only.
    """
    if random.random() > 0.5:
        frame      = TF.hflip(frame)
        prev_masks = [TF.hflip(m) for m in prev_masks]
        mask       = TF.hflip(mask)

    if random.random() > 0.5:
        frame      = TF.vflip(frame)
        prev_masks = [TF.vflip(m) for m in prev_masks]
        mask       = TF.vflip(mask)

    angle  = random.uniform(-15, 15)
    frame      = TF.rotate(frame, angle)
    prev_masks = [TF.rotate(m, angle) for m in prev_masks]
    mask       = TF.rotate(mask, angle)

    # Brightness jitter — frame only
    factor = random.uniform(0.8, 1.2)
    frame  = torch.clamp(frame * factor, 0.0, 1.0)

    return frame, prev_masks, mask


# ── Scheduled sampling corruption ─────────────────────────────────────────────

def _corrupt_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Randomly corrupt the previous mask to discourage shortcut learning.

    Without corruption the network learns to copy the conditioning mask
    rather than consult the image, achieving high teacher-forced Dice
    but remaining static under autoregressive deployment.

    Corruption probabilities (applied during training only):
        0.50 — no corruption (ground truth)
        0.20 — spatial shift ±4 pixels
        0.15 — Gaussian noise (sigma=0.15)
        0.15 — blank mask (all zeros)
    """
    r = random.random()
    if r < 0.50:
        return mask                                        # ground truth
    elif r < 0.70:
        dx = random.randint(-4, 4)
        dy = random.randint(-4, 4)
        return torch.roll(torch.roll(mask, dx, dims=-1), dy, dims=-2)
    elif r < 0.85:
        noise = torch.randn_like(mask) * 0.15
        return torch.clamp(mask + noise, 0.0, 1.0)
    else:
        return torch.zeros_like(mask)                      # blank


# ── Pre-loading dataset ───────────────────────────────────────────────────────

class TrackRADDataset(Dataset):
    """
    Pre-loading dataset — reads all .mha files at initialisation and
    stores frames and masks as float16 tensors in memory.

    Suitable for N=1. For N>=3 on machines with <32GB RAM, use
    TrackRADLazyDataset instead.

    Args:
        root_dir:      path to trackrad2025_labeled_training_data_50/
        patient_ids:   list of patient folder names (e.g. ['A_001', ...])
        augment:       whether to apply data augmentation
        n_prev_masks:  number of previous masks to stack (N)
    """

    def __init__(self,
                 root_dir: str,
                 patient_ids: list,
                 augment: bool = False,
                 n_prev_masks: int = 1):
        self.samples      = []
        self.augment      = augment
        self.n_prev_masks = n_prev_masks
        loaded, skipped   = [], []

        for pid in patient_ids:
            frames_path, labels_path = find_patient_files(root_dir, pid)
            if frames_path is None:
                skipped.append(pid)
                continue

            try:
                frames = sitk.GetArrayFromImage(sitk.ReadImage(frames_path))
                labels = sitk.GetArrayFromImage(sitk.ReadImage(labels_path))
            except Exception as e:
                print(f"  Error reading {pid}: {e}")
                skipped.append(pid)
                continue

            # (H, W, T) -> (T, H, W)
            frames = np.transpose(frames, (2, 0, 1)).astype(np.float32)
            labels = np.transpose(labels, (2, 0, 1)).astype(np.float32)
            frames = frames / (frames.max() + 1e-8)

            T = frames.shape[0]
            for t in range(T):
                frame_t    = _resize_frame(frames[t])
                mask_t     = _resize_mask(labels[t])
                prev_masks = [
                    _resize_mask(labels[max(t - k, 0)])
                    for k in range(1, n_prev_masks + 1)
                ]
                # Store as float16 to halve memory footprint
                self.samples.append((
                    frame_t.numpy().astype(np.float16),
                    [m.numpy().astype(np.float16) for m in prev_masks],
                    mask_t.numpy().astype(np.float16),
                ))
            loaded.append(pid)

        print(f"[TrackRADDataset n_prev={n_prev_masks}] "
              f"Loaded {len(self.samples)} samples from {len(loaded)} patients")
        if skipped:
            print(f"  Skipped: {skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame_t, prev_masks, mask_t = self.samples[idx]
        # Cast back to float32 for network input
        frame_t    = torch.tensor(frame_t.astype(np.float32))
        prev_masks = [torch.tensor(m.astype(np.float32)) for m in prev_masks]
        mask_t     = torch.tensor(mask_t.astype(np.float32))

        if self.augment:
            frame_t, prev_masks, mask_t = _augment(frame_t, prev_masks, mask_t)
            prev_masks[0] = _corrupt_mask(prev_masks[0])

        return torch.cat([frame_t] + prev_masks, dim=0), mask_t


# ── Lazy loading dataset ──────────────────────────────────────────────────────

class TrackRADLazyDataset(Dataset):
    """
    Lazy loading dataset — stores only file paths and frame indices.
    Reads .mha files on demand in __getitem__ with a per-patient cache.

    RAM usage is approximately 2x the size of one patient's data,
    regardless of N. Suitable for all N values on all machines.

    Args: same as TrackRADDataset.
    """

    def __init__(self,
                 root_dir: str,
                 patient_ids: list,
                 augment: bool = False,
                 n_prev_masks: int = 1):
        self.root_dir     = root_dir
        self.augment      = augment
        self.n_prev_masks = n_prev_masks
        self.index        = []   # (frames_path, labels_path, t, T)
        self._cache       = {}   # {path: ndarray} — per-patient cache
        loaded, skipped   = [], []

        for pid in patient_ids:
            frames_path, labels_path = find_patient_files(root_dir, pid)
            if frames_path is None:
                skipped.append(pid)
                continue
            # Read only the size — do not load pixel data yet
            img = sitk.ReadImage(frames_path)
            T   = img.GetSize()[2]
            for t in range(T):
                self.index.append((frames_path, labels_path, t, T))
            loaded.append(pid)

        print(f"[TrackRADLazyDataset n_prev={n_prev_masks}] "
              f"{len(self.index)} samples from {len(loaded)} patients")
        if skipped:
            print(f"  Skipped: {skipped}")

    def _load(self, path: str) -> np.ndarray:
        """Load and cache a .mha file (frames or labels)."""
        if path not in self._cache:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
            arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
            if "frames" in path:
                arr = arr / (arr.max() + 1e-8)
            self._cache[path] = arr
        return self._cache[path]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        frames_path, labels_path, t, T = self.index[idx]

        frames = self._load(frames_path)
        labels = self._load(labels_path)

        frame_t    = _resize_frame(frames[t])
        mask_t     = _resize_mask(labels[t])
        prev_masks = [
            _resize_mask(labels[max(t - k, 0)])
            for k in range(1, self.n_prev_masks + 1)
        ]

        if self.augment:
            frame_t, prev_masks, mask_t = _augment(frame_t, prev_masks, mask_t)
            prev_masks[0] = _corrupt_mask(prev_masks[0])

        return torch.cat([frame_t] + prev_masks, dim=0), mask_t
