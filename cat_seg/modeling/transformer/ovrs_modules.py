"""Small, class-shared modules for CAT-Seg ablations.

Mechanisms follow LSKNet, PCA-Seg and GeoSeg-OV; this is a local
reimplementation, not a reproduction of their complete systems.
"""
import torch
from torch import nn


def group_norm(channels):
    groups = min(32, max(channels // 4, 1))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class LSKGuidanceAdapter(nn.Module):
    """Process B shared guidance maps once, before category broadcasting."""

    def __init__(self, channels, dilation=3, residual_scale=0.01):
        super().__init__()
        if channels < 2 or channels % 2 or dilation < 1:
            raise ValueError("LSK requires even channels >= 2 and dilation >= 1")
        self.pre = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU())
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.context = nn.Conv2d(
            channels, channels, 7, padding=3 * dilation,
            dilation=dilation, groups=channels,
        )
        self.local_reduce = nn.Conv2d(channels, channels // 2, 1)
        self.context_reduce = nn.Conv2d(channels, channels // 2, 1)
        self.selector = nn.Conv2d(2, 2, 7, padding=3)
        self.expand = nn.Conv2d(channels // 2, channels, 1)
        self.post = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, guidance):
        feature = self.pre(guidance)
        local = self.local(feature)
        context = self.context(local)
        local = self.local_reduce(local)
        context = self.context_reduce(context)
        pair = torch.cat([local, context], dim=1)
        statistics = torch.cat([
            pair.mean(dim=1, keepdim=True), pair.amax(dim=1, keepdim=True)
        ], dim=1)
        weights = self.selector(statistics).sigmoid()
        selected = weights[:, :1] * local + weights[:, 1:] * context
        update = self.post(feature * self.expand(selected))
        return guidance + self.scale * update


class ConvGuidanceAdapter(nn.Module):
    """Simple spatial adaptation control; not parameter-matched to LSK."""

    def __init__(self, channels, residual_scale=0.01):
        super().__init__()
        self.update = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.GELU(), nn.Conv2d(channels, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, guidance):
        return guidance + self.scale * self.update(guidance)


class ParallelFusion(nn.Module):
    """Fuse [B,C,K,H,W] branch outputs with parameters shared across K.

EPL adaptation uses GroupNorm instead of source SyncBatchNorm. This avoids
coupling normalization statistics to the current category set / GPU batch.
"""

    def __init__(self, channels, mode="simple", experts=4, residual_scale=0.01):
        super().__init__()
        if mode not in {"simple", "epl"} or experts < 1:
            raise ValueError("fusion mode must be simple/epl; experts must be positive")
        self.mode = mode
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))
        if mode == "simple":
            self.fuse = nn.Conv2d(2 * channels, channels, 1)
        else:
            self.spatial_proj = nn.Conv2d(channels, channels, 1)
            self.class_proj = nn.Conv2d(channels, channels, 1)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(2 * channels, 2 * channels, 1),
                    group_norm(2 * channels), nn.GELU(),
                    nn.Conv2d(2 * channels, channels, 3, padding=1, groups=channels),
                    nn.GELU(),
                ) for _ in range(experts)
            ])
            self.gate = nn.Sequential(
                nn.Conv2d(2 * channels, max(channels // 4, 1), 1), nn.ReLU(),
                nn.Conv2d(max(channels // 4, 1), experts, 1), nn.Softmax(dim=1),
            )
            self.residual = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, spatial, category):
        if spatial.shape != category.shape or spatial.ndim != 5:
            raise ValueError("parallel branches must have the same [B,C,K,H,W] shape")
        batch, channels, classes, height, width = spatial.shape
        spatial = spatial.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        category = category.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        base = 0.5 * (spatial + category)
        if self.mode == "simple":
            fused = self.fuse(torch.cat([spatial, category], dim=1))
        else:
            feature = torch.cat([self.spatial_proj(spatial), self.class_proj(category)], dim=1)
            weights = self.gate(feature)
            fused = self.residual(feature)
            for index, expert in enumerate(self.experts):
                fused = fused + weights[:, index:index + 1] * expert(feature)
        fused = base + self.scale * fused
        return fused.reshape(batch, classes, channels, height, width).permute(0, 2, 1, 3, 4)


class CostGuidanceRefiner(nn.Module):
    """Single-stream CAD; summary is a decoder feature, not confidence."""

    def __init__(self, cost_channels, guidance_channels, residual_scale=0.01):
        super().__init__()
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

    def forward(self, upsampled_cost, guidance):
        if guidance.ndim != 4 or upsampled_cost.ndim != 4:
            raise ValueError("CAD expects [B*K,C,H,W] cost and [B,D,H,W] guidance")
        batch = guidance.shape[0]
        if batch < 1 or upsampled_cost.shape[0] % batch:
            raise ValueError("cost batch must be an integer multiple of guidance batch")
        if upsampled_cost.shape[-2:] != guidance.shape[-2:]:
            raise ValueError("CAD cost and guidance spatial sizes must match")
        summary = upsampled_cost.reshape(batch, -1, *upsampled_cost.shape[1:]).mean(dim=1)
        relevance = (self.cost_projection(summary) * guidance).sigmoid()
        return guidance + self.scale * self.refinement(relevance * guidance)
