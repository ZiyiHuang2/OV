"""Loss shared by every ablation, including the serial baseline."""
import torch
from torch.nn import functional as F


def masked_segmentation_bce(logits, targets, ignore_index=255):
    if logits.ndim != 4 or targets.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError('Expected [B,K,H,W] logits and [B,H,W] target')
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    labels = targets[valid]
    classes = logits.shape[1]
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError('Mask IDs must match the contiguous training vocabulary')
    selected_logits = logits.permute(0, 2, 3, 1)[valid]
    binary_targets = F.one_hot(labels.long(), classes).to(selected_logits.dtype)
    return F.binary_cross_entropy_with_logits(selected_logits, binary_targets)
