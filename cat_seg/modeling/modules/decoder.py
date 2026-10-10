"""M3: shared or class-conditioned CAD-inspired guidance refinement."""
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

    @staticmethod
    def _validate(upsampled_cost, guidance):
        if guidance.ndim != 4 or upsampled_cost.ndim != 4:
            raise ValueError("CAD expects [B*K,C,H,W] cost and [B,D,H,W] guidance")
        batch = guidance.shape[0]
        if batch < 1 or upsampled_cost.shape[0] < 1 or upsampled_cost.shape[0] % batch:
            raise ValueError("cost batch must be a positive integer multiple of guidance batch")
        if upsampled_cost.shape[-2:] != guidance.shape[-2:]:
            raise ValueError("CAD cost and guidance spatial sizes must match")
        return batch, upsampled_cost.shape[0] // batch

    def _refine(self, cost, guidance):
        relevance = (self.cost_projection(cost) * guidance).sigmoid()
        return guidance + self.scale * self.refinement(relevance * guidance)

    def forward(self, upsampled_cost, guidance):
        batch, _ = self._validate(upsampled_cost, guidance)
        summary = self.summarize(upsampled_cost, batch)
        return self._refine(summary, guidance)


class ClassGuidanceRefiner(CostGuidanceRefiner):
    """Apply the same CAD transform independently to each real candidate class.

    Inputs retain the shared CAD contract: cost [B*K,C,H,W], guidance [B,D,H,W].
    Output is [B*K,D,H,W] in image-major, class-minor order. No class summary,
    class-specific parameters or fixed vocabulary size are introduced. GroupNorm
    operates on each (image, class) sample separately.
    """

    def __init__(self, cost_channels, guidance_channels, residual_scale=0.01,
                 pooling="mean"):
        if pooling != "mean":
            raise ValueError("per_class CAD requires CAD_POOLING mean; it does not pool classes")
        super().__init__(cost_channels, guidance_channels, residual_scale, pooling)

    def forward(self, upsampled_cost, guidance):
        batch, classes = self._validate(upsampled_cost, guidance)
        guidance = guidance[:, None].expand(batch, classes, *guidance.shape[1:])
        guidance = guidance.reshape(batch * classes, *guidance.shape[2:])
        return self._refine(upsampled_cost, guidance)
