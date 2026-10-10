"""M2: simple, EPL-inspired and selective fusion of parallel branches."""
import torch
from torch import nn

from .common import group_norm


class ParallelFusion(nn.Module):
    """Fuse [B,C,K,H,W] branch outputs with parameters shared across K.

EPL adaptation uses GroupNorm instead of source SyncBatchNorm. This avoids
coupling normalization statistics to the current category set / GPU batch.
"""

    def __init__(self, channels, mode="simple", experts=4, residual_scale=0.01,
                 epl_routing="learned"):
        super().__init__()
        if mode not in {"simple", "epl", "gated"} or experts < 1:
            raise ValueError("fusion mode must be simple/epl/gated; experts must be positive")
        if epl_routing not in {"learned", "uniform"}:
            raise ValueError("EPL routing must be learned/uniform")
        if mode != "epl" and epl_routing != "learned":
            raise ValueError("uniform EPL routing requires EPL fusion")
        self.mode = mode
        self.epl_routing = epl_routing
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))
        if mode == "simple":
            self.fuse = nn.Conv2d(2 * channels, channels, 1)
        elif mode == "gated":
            # Each class/location chooses between two different aggregation paths.
            # These scores are learned preferences, not calibrated uncertainty.
            self.route = nn.Sequential(
                group_norm(3 * channels),
                nn.Conv2d(3 * channels, max(channels // 4, 1), 1), nn.GELU(),
                nn.Conv2d(max(channels // 4, 1), 2, 1),
            )
            nn.init.normal_(self.route[-1].weight, std=1e-3)
            nn.init.zeros_(self.route[-1].bias)
            self.refine = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1, bias=False),
                group_norm(channels), nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            )
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
            if epl_routing == "uniform":
                # Keep construction order and state keys identical to learned EPL.
                # The router is stored, but never executed or optimized.
                self.gate.requires_grad_(False)

    def forward(self, spatial, category):
        if spatial.shape != category.shape or spatial.ndim != 5:
            raise ValueError("parallel branches must have the same [B,C,K,H,W] shape")
        batch, channels, classes, height, width = spatial.shape
        spatial = spatial.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        category = category.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        base = 0.5 * (spatial + category)
        if self.mode == "simple":
            fused = self.fuse(torch.cat([spatial, category], dim=1))
        elif self.mode == "gated":
            routing = torch.cat([spatial, category, (spatial - category).abs()], dim=1)
            weights = self.route(routing).float().softmax(dim=1).to(spatial.dtype)
            base = weights[:, :1] * spatial + weights[:, 1:] * category
            fused = self.refine(torch.cat([spatial, category], dim=1))
        else:
            feature = torch.cat([self.spatial_proj(spatial), self.class_proj(category)], dim=1)
            weights = self.gate(feature) if self.epl_routing == "learned" else None
            fused = self.residual(feature)
            for index, expert in enumerate(self.experts):
                weight = weights[:, index:index + 1] if weights is not None else 1.0 / len(self.experts)
                fused = fused + weight * expert(feature)
        fused = base + self.scale * fused
        return fused.reshape(batch, classes, channels, height, width).permute(0, 2, 1, 3, 4)
