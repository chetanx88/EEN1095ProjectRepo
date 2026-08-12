"""
Loss Functions and Evaluation Metrics
EEN1095 Implementation Project — TrackRAD2025

Author: Chetan Kumar (A00054853)
Dublin City University, August 2026

Loss functions:
    focal_loss       — handles class imbalance (tumour ~1-2% of frame)
    tversky_loss     — penalises false negatives > false positives
    combined_loss    — focal + tversky (training objective)

Evaluation metrics (official TrackRAD2025):
    dice_score       — Dice similarity coefficient
    surface_dist_95  — 95th percentile symmetric surface distance (px)
    com_distance     — 2D centre-of-mass distance (px)
    dosimetric_acc   — fraction of tumour covered by dilated MLC beam
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation


# ── Loss functions ────────────────────────────────────────────────────────────

def focal_loss(pred: torch.Tensor,
               target: torch.Tensor,
               alpha: float = 0.8,
               gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss — down-weights easy background pixels so gradient
    signal concentrates on the small tumour region.

    Args:
        pred:   raw logits (B, 1, H, W)
        target: binary ground truth (B, 1, H, W)
        alpha:  weighting factor
        gamma:  focusing parameter
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt  = torch.exp(-bce)
    return (alpha * (1.0 - pt) ** gamma * bce).mean()


def tversky_loss(pred: torch.Tensor,
                 target: torch.Tensor,
                 alpha: float = 0.3,
                 beta: float  = 0.7,
                 smooth: float = 1e-6) -> torch.Tensor:
    """
    Tversky loss — asymmetric generalisation of Dice.
    beta > alpha penalises false negatives more than false positives,
    counteracting under-segmentation of small tumour volumes.

    Args:
        pred:   raw logits (B, 1, H, W)
        target: binary ground truth (B, 1, H, W)
        alpha:  false positive weight
        beta:   false negative weight (set > alpha to penalise FN)
    """
    pred = torch.sigmoid(pred)
    tp   = (pred * target).sum(dim=(2, 3))
    fp   = (pred * (1.0 - target)).sum(dim=(2, 3))
    fn   = ((1.0 - pred) * target).sum(dim=(2, 3))
    return (1.0 - ((tp + smooth) /
                   (tp + alpha * fp + beta * fn + smooth))).mean()


def combined_loss(pred: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    """Training objective: focal + tversky."""
    return focal_loss(pred, target) + tversky_loss(pred, target)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor,
               target: torch.Tensor,
               smooth: float = 1e-6) -> float:
    """
    Dice similarity coefficient.

    Args:
        pred:   raw logits (B, 1, H, W)
        target: binary ground truth (B, 1, H, W)

    Returns:
        Mean Dice over batch as a Python float.
    """
    pred  = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    return ((2.0 * inter + smooth) /
            (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
            ).mean().item()


def surface_dist_95(pred_mask: np.ndarray,
                    gt_mask: np.ndarray) -> float:
    """
    95th percentile symmetric surface distance in pixels.

    Boundaries are extracted by subtracting the binary erosion from
    the mask (one-pixel erosion with a 3x3 structuring element).
    The distance transform is computed from each boundary to the other,
    and the 95th percentile of the combined set is returned.

    Args:
        pred_mask: binary numpy array (H, W), uint8 or bool
        gt_mask:   binary numpy array (H, W), uint8 or bool

    Returns:
        SD95 in pixels, or float('nan') if either mask is empty.
    """
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")

    def get_boundary(mask):
        eroded = binary_erosion(mask.astype(bool))
        return mask.astype(bool) & ~eroded

    pred_b = get_boundary(pred_mask)
    gt_b   = get_boundary(gt_mask)

    if not pred_b.any() or not gt_b.any():
        return float("nan")

    # Distance from pred boundary to gt boundary and vice versa
    d1    = distance_transform_edt(~gt_b)[pred_b]
    d2    = distance_transform_edt(~pred_b)[gt_b]
    all_d = np.concatenate([d1, d2])

    return float(np.percentile(all_d, 95)) if len(all_d) > 0 else float("nan")


def com_distance(pred_mask: np.ndarray,
                 gt_mask: np.ndarray) -> float:
    """
    2D centre-of-mass distance in pixels.

    Args:
        pred_mask: binary numpy array (H, W)
        gt_mask:   binary numpy array (H, W)

    Returns:
        Euclidean distance between centroids in pixels,
        or float('nan') if either mask is empty.
    """
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")

    def centroid(m):
        idx = np.argwhere(m.astype(bool))
        return idx.mean(axis=0)   # (row, col)

    c_pred = centroid(pred_mask)
    c_gt   = centroid(gt_mask)
    return float(np.sqrt(((c_pred - c_gt) ** 2).sum()))


def dosimetric_acc(pred_mask: np.ndarray,
                   gt_mask: np.ndarray,
                   margin_px: int = 3) -> float:
    """
    Dosimetric accuracy — fraction of tumour covered by the MLC beam.

    Simulates an MLC aperture by dilating the predicted mask with a
    square structuring element of side (2*margin_px + 1). The metric
    measures what fraction of the reference tumour falls within the beam.
    A margin of 3 pixels corresponds to approximately 5mm at typical
    cine-MRI resolution.

    Args:
        pred_mask:  binary numpy array (H, W)
        gt_mask:    binary numpy array (H, W)
        margin_px:  MLC margin in pixels

    Returns:
        Dosimetric accuracy in [0, 1], or float('nan') if empty.
    """
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")

    struct = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), dtype=bool)
    beam   = binary_dilation(pred_mask.astype(bool), structure=struct)
    covered = (beam & gt_mask.astype(bool)).sum()
    return float(covered / gt_mask.sum())


# ── Batch evaluation helper ───────────────────────────────────────────────────

def evaluate_batch(pred_logits: torch.Tensor,
                   gt_masks: torch.Tensor) -> dict:
    """
    Compute all four metrics for a batch, skipping empty masks.

    Args:
        pred_logits: (B, 1, H, W) raw logits
        gt_masks:    (B, 1, H, W) binary ground truth

    Returns:
        dict with keys: dice, sd95, com, dos
        Values are batch means (NaN frames excluded).
    """
    preds_bin = (torch.sigmoid(pred_logits) > 0.5).cpu().numpy().astype(np.uint8)
    gts       = gt_masks.cpu().numpy().astype(np.uint8)
    B         = preds_bin.shape[0]

    results = {"dice": [], "sd95": [], "com": [], "dos": []}

    for b in range(B):
        p = preds_bin[b, 0]
        g = gts[b, 0]
        if g.sum() == 0:
            continue
        results["dice"].append(
            dice_score(pred_logits[b:b+1], gt_masks[b:b+1]))
        results["sd95"].append(surface_dist_95(p, g))
        results["com"].append(com_distance(p, g))
        results["dos"].append(dosimetric_acc(p, g))

    def safe_mean(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {k: safe_mean(v) for k, v in results.items()}
