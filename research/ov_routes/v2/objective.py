"""Calibrated gain regression plus cost-sensitive pairwise window ranking."""
import torch
from torch.nn import functional as F


def pairwise_ranking(pred: torch.Tensor, true: torch.Tensor, min_gap: float = 0.02):
    """Only order nontrivially different crop gains; no GT used at inference.

    Inputs are [B,A] in scaled BCE units (raw BCE gain * 100). The cutoff
    prevents almost tied oracle actions from dominating with arbitrary labels.
    """
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError('pred/true need matching [B,A] shape')
    tdiff = true.unsqueeze(2) - true.unsqueeze(1)
    pdiff = pred.unsqueeze(2) - pred.unsqueeze(1)
    upper = torch.triu(torch.ones_like(tdiff,dtype=torch.bool),diagonal=1)
    eligible = upper & (tdiff.abs() > min_gap)
    if not eligible.any():
        return pred.sum() * 0
    signs = (tdiff[eligible] > 0).to(pred.dtype)
    weights = tdiff[eligible].abs().detach().clamp(max=2.)
    loss = F.binary_cross_entropy_with_logits(pdiff[eligible],signs,reduction='none')
    return (weights*loss).sum()/weights.sum().clamp_min(1e-12)


def selector_loss(pred: torch.Tensor, gains: torch.Tensor,
                  gain_scale: float = 100.0, regression_weight: float = 1.0,
                  ranking_weight: float = 1.0):
    """Predictions express scaled BCE improvement; 0 has no-op meaning."""
    target = gains*gain_scale
    mse = F.mse_loss(pred,target)
    ranking = pairwise_ranking(pred,target)
    total = regression_weight*mse + ranking_weight*ranking
    return total, {'mse':float(mse.detach()),'rank':float(ranking.detach())}
