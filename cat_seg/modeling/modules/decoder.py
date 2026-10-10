"""M3: single-stream CAD-inspired guidance refinement."""
from torch import nn
import torch

from .common import group_norm


class CostGuidanceRefiner(nn.Module):
    """Single-stream CAD; summary is a decoder feature, not confidence."""

    def __init__(self, cost_channels, guidance_channels, residual_scale=0.01, pooling="mean"):
        super().__init__()
        if pooling not in {"mean", "weighted"}:
            raise ValueError("CAD pooling must be mean/weighted")
        self.class_score = nn.Conv2d(cost_channels, 1, 1, bias=False) if pooling == "weighted" else None
        if self.class_score is not None:
            # Start near uniform pooling while retaining gradients through the scorer.
            nn.init.normal_(self.class_score.weight, std=1e-3)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.cost_projection = nn.Sequential(
            nn.Conv2d(cost_channels, guidance_channels, 1, bias=False),
            group_norm(guidance_channels),
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(guidance_channels, guidance_channels, 3, padding=1,
                      groups=guidance_channels, bias=False),
            nn.Conv2d(guidance_channels, guidance_channels, 1, bias=False),
            group_norm(guidance_channels),
        )

    def summarize(self, upsampled_cost, batch):
        cost = upsampled_cost.reshape(batch, -1, *upsampled_cost.shape[1:])
        if self.class_score is None:
            return cost.mean(dim=1)
        scores = self.class_score(upsampled_cost).reshape(batch, -1, 1, *cost.shape[-2:])
        weights = scores.float().softmax(dim=1).to(cost.dtype)
        return (weights * cost).sum(dim=1)

    def forward(self, upsampled_cost, guidance):
        if guidance.ndim != 4 or upsampled_cost.ndim != 4:
            raise ValueError("CAD expects [B*K,C,H,W] cost and [B,D,H,W] guidance")
        batch = guidance.shape[0]
        if batch < 1 or upsampled_cost.shape[0] % batch:
            raise ValueError("cost batch must be an integer multiple of guidance batch")
        if upsampled_cost.shape[-2:] != guidance.shape[-2:]:
            raise ValueError("CAD cost and guidance spatial sizes must match")
        summary = self.summarize(upsampled_cost, batch)
        relevance = (self.cost_projection(summary) * guidance).sigmoid()
        return guidance + self.scale * self.refinement(relevance * guidance)
