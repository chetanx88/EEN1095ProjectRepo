
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def dice_loss(pred, target, smooth=1e-6):
    pred  = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    return 1 - ((2 * inter + smooth) /
                (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)).mean()

def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1e-6):
    pred = torch.sigmoid(pred)
    tp   = (pred * target).sum(dim=(2, 3))
    fp   = (pred * (1 - target)).sum(dim=(2, 3))
    fn   = ((1 - pred) * target).sum(dim=(2, 3))
    return 1 - ((tp + smooth) / (tp + alpha*fp + beta*fn + smooth)).mean()

def combined_loss(pred, target):
    return focal_loss(pred, target) + tversky_loss(pred, target)

def dice_score(pred, target, smooth=1e-6):
    pred  = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    return ((2 * inter + smooth) /
            (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)).mean().item()

def surface_distance_95(pred_mask, gt_mask):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")
    def get_boundary(mask):
        eroded = binary_erosion(mask.astype(bool))
        return mask.astype(bool) & ~eroded
    pred_b = get_boundary(pred_mask)
    gt_b   = get_boundary(gt_mask)
    if not pred_b.any() or not gt_b.any():
        return float("nan")
    d1 = distance_transform_edt(~gt_b)[pred_b]
    d2 = distance_transform_edt(~pred_b)[gt_b]
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95)) if len(all_d) > 0 else float("nan")

def centre_of_mass_distance(pred_mask, gt_mask):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")
    def centroid(m):
        idx = np.argwhere(m); return idx.mean(axis=0)
    return float(np.sqrt(((centroid(pred_mask) - centroid(gt_mask)) ** 2).sum()))

def dosimetric_accuracy(pred_mask, gt_mask, margin_pixels=3):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")
    from scipy.ndimage import binary_dilation
    beam  = binary_dilation(pred_mask.astype(bool), np.ones((margin_pixels*2+1, margin_pixels*2+1)))
    return float((beam & gt_mask.astype(bool)).sum() / gt_mask.sum())
