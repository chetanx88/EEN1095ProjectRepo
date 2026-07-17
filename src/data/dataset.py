
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
import os

TARGET_SIZE = 256

def find_patient_files(root_dir, folder_name):
    patient_path = os.path.join(root_dir, folder_name)
    if not os.path.exists(patient_path):
        return None, None
    images_dir  = os.path.join(patient_path, "images")
    targets_dir = os.path.join(patient_path, "targets")
    if os.path.exists(images_dir) and os.path.exists(targets_dir):
        frame_files = [f for f in os.listdir(images_dir) if f.endswith("_frames.mha")]
        label_files = [f for f in os.listdir(targets_dir) if f.endswith("_labels.mha")]
        if frame_files and label_files:
            return (os.path.join(images_dir, frame_files[0]),
                    os.path.join(targets_dir, label_files[0]))
    return None, None


class TrackRADDataset(Dataset):
    def __init__(self, root_dir, patient_ids, augment=False, n_prev_masks=1):
        self.samples      = []
        self.augment      = augment
        self.n_prev_masks = n_prev_masks
        loaded, skipped   = [], []

        for pid in patient_ids:
            frames_path, labels_path = find_patient_files(root_dir, pid)
            if frames_path is None:
                skipped.append(pid); continue

            frames = sitk.GetArrayFromImage(sitk.ReadImage(frames_path))
            labels = sitk.GetArrayFromImage(sitk.ReadImage(labels_path))

            frames = np.transpose(frames, (2, 0, 1)).astype(np.float32)
            labels = np.transpose(labels, (2, 0, 1)).astype(np.float32)
            frames = frames / (frames.max() + 1e-8)

            T = frames.shape[0]
            for t in range(T):
                frame_t = self._resize(frames[t], "bilinear")
                mask_t  = self._resize(labels[t], "nearest")
                prev_masks = []
                for k in range(1, n_prev_masks + 1):
                    prev_idx = max(t - k, 0)
                    prev_masks.append(self._resize(labels[prev_idx], "nearest"))
                self.samples.append((
                    frame_t.numpy().astype(np.float16),
                    [m.numpy().astype(np.float16) for m in prev_masks],
                    mask_t.numpy().astype(np.float16)
                ))
            loaded.append(pid)

        print(f"[n_prev_masks={n_prev_masks}] Loaded {len(self.samples)} samples from {len(loaded)} patients")
        if skipped:
            print(f"Skipped: {skipped}")

    def _resize(self, arr, mode):
        t  = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
        kw = {"align_corners": False} if mode == "bilinear" else {}
        return F.interpolate(t, size=(TARGET_SIZE, TARGET_SIZE), mode=mode, **kw).squeeze(0)

    def __len__(self):
        return len(self.samples)

    def _augment(self, frame, prev_masks, mask):
        if random.random() > 0.5:
            frame = TF.hflip(frame)
            prev_masks = [TF.hflip(m) for m in prev_masks]
            mask  = TF.hflip(mask)
        if random.random() > 0.5:
            frame = TF.vflip(frame)
            prev_masks = [TF.vflip(m) for m in prev_masks]
            mask  = TF.vflip(mask)
        angle = random.uniform(-15, 15)
        frame = TF.rotate(frame, angle)
        prev_masks = [TF.rotate(m, angle) for m in prev_masks]
        mask  = TF.rotate(mask, angle)
        return frame, prev_masks, mask

    def __getitem__(self, idx):
        frame_t, prev_masks, mask_t = self.samples[idx]
        frame_t    = torch.tensor(frame_t.astype(np.float32))
        prev_masks = [torch.tensor(m.astype(np.float32)) for m in prev_masks]
        mask_t     = torch.tensor(mask_t.astype(np.float32))
        if self.augment:
            frame_t, prev_masks, mask_t = self._augment(frame_t, prev_masks, mask_t)
        x = torch.cat([frame_t] + prev_masks, dim=0)
        return x, mask_t
