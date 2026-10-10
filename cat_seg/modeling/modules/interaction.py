"""Experimental cost-conditioned interactions; not upstream reproductions.

Design references and adaptation boundaries are recorded in docs/INTERACTION.md.
All transforms share parameters across classes and normalize each image/class
separately. No fixed vocabulary, extra teacher, or auxiliary loss is required.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import group_norm
from .decoder import ClassGuidanceRefiner


class GatedSpatialFFN(nn.Module):
    """Expanded channels with multiplicative gating (Restormer-inspired)."""

    def __init__(self, channels, spatial="local"):
        super().__init__()
        hidden = 2 * channels
        self.norm = group_norm(channels)
        self.expand = nn.Conv2d(channels, 2 * hidden, 1)
        width = 2 * hidden
        if spatial == "local":
            self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        elif spatial == "context":
            self.spatial = nn.Sequential(
                nn.Conv2d(width, width, 5, padding=2, groups=width),
                nn.Conv2d(width, width, 7, padding=6, dilation=2, groups=width),
            )
        elif spatial == "strip":
            self.spatial = nn.Sequential(
                nn.Conv2d(width, width, (1, 7), padding=(0, 3), groups=width),
                nn.Conv2d(width, width, (7, 1), padding=(3, 0), groups=width),
            )
        elif spatial == "channel":
            self.spatial = nn.Identity()
        else:
            raise ValueError("unknown spatial expert: " + spatial)
        self.project = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        content, gate = self.spatial(self.expand(self.norm(x))).chunk(2, dim=1)
        return self.project(F.gelu(content) * gate)


class CostConditionedGuidance(nn.Module):
    """Reselect shared spatial context from each layer's real cost volume.

    This module lives inside AggregatorLayer. Every call anchors to the original
    projected guidance, rather than repeatedly accumulating guidance residuals.
    The learned class summary is a feature preference, not class confidence.
    """

    def __init__(self, cost_channels, guidance_channels, dilation=3, residual_scale=0.01):
        super().__init__()
        if guidance_channels < 2 or dilation < 1:
            raise ValueError("cost LSK requires guidance channels >= 2 and dilation >= 1")
        dim = guidance_channels
        self.pre = nn.Sequential(group_norm(dim), nn.Conv2d(dim, dim, 1), nn.GELU())
        self.local = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.context = nn.Conv2d(dim, dim, 7, padding=3 * dilation,
                                 dilation=dilation, groups=dim)
        self.strip = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
        )
        self.cost_projection = nn.Sequential(nn.Conv2d(cost_channels, dim, 1), group_norm(dim))
        # A class-shared additive bias cancels in the class softmax.
        self.class_score = nn.Conv2d(dim, 1, 1, bias=False)
        self.selector = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, 3, 1),
        )
        self.post = nn.Conv2d(dim, dim, 1)
        self.ffn = GatedSpatialFFN(dim)
        self.scale = nn.Parameter(torch.full((1, dim, 1, 1), float(residual_scale)))
        self.ffn_scale = nn.Parameter(torch.full((1, dim, 1, 1), float(residual_scale)))

    def forward(self, guidance, cost):
        if guidance is None or cost.ndim != 5 or guidance.ndim != 4:
            raise ValueError("cost LSK expects cost [B,C,K,H,W] and guidance [B,D,H,W]")
        batch, channels, classes, height, width = cost.shape
        if classes < 1 or guidance.shape[0] != batch or guidance.shape[-2:] != (height, width):
            raise ValueError("cost LSK guidance and real cost dimensions must agree")
        feature = self.pre(guidance)
        projected = self.cost_projection(cost.permute(0, 2, 1, 3, 4).reshape(
            batch * classes, channels, height, width))
        guide = feature[:, None].expand(-1, classes, -1, -1, -1).reshape_as(projected)
        scores = self.class_score(F.gelu(projected * guide)).reshape(batch, classes, 1, height, width)
        weights = scores.float().softmax(dim=1).to(projected.dtype)
        summary = (weights * projected.reshape(batch, classes, -1, height, width)).sum(dim=1)
        local = self.local(feature)
        banks = torch.stack([local, self.context(local), self.strip(feature)], dim=1)
        gates = self.selector(torch.cat([feature, summary], dim=1)).sigmoid()
        selected = (banks * gates[:, :, None]).sum(dim=1)
        updated = guidance + self.scale * self.post(feature * selected)
        return updated + self.ffn_scale * self.ffn(updated)


class InteractionFusion(nn.Module):
    """Dense heterogeneous experts with branch/visual/text conditioned routing.

    Architectural roles are different; learned semantic specialization remains
    an experimental question. This is not sparse MoE or a faithful EPL copy.
    """

    def __init__(self, channels, guidance_channels, text_channels, residual_scale=0.01):
        super().__init__()
        if min(channels, guidance_channels, text_channels) < 1:
            raise ValueError("interaction fusion requires cost, visual and text guidance")
        self.input_projection = nn.Sequential(
            group_norm(4 * channels), nn.Conv2d(4 * channels, channels, 1), nn.GELU(),
        )
        self.visual_projection = nn.Conv2d(guidance_channels, channels, 1)
        self.text_projection = nn.Linear(text_channels, channels)
        self.router = nn.Sequential(
            group_norm(3 * channels), nn.Conv2d(3 * channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, 4, 1),
        )
        self.experts = nn.ModuleList([
            GatedSpatialFFN(channels, spatial) for spatial in ("local", "context", "strip", "channel")
        ])
        self.refine = GatedSpatialFFN(channels)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), float(residual_scale)))
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(residual_scale)))

    def forward(self, spatial, category, guidance, text):
        if spatial.ndim != 5 or spatial.shape != category.shape:
            raise ValueError("interaction branches must have equal [B,C,K,H,W] shapes")
        batch, channels, classes, height, width = spatial.shape
        if classes < 1 or guidance is None or text is None:
            raise ValueError("interaction fusion needs real classes and both guidance streams")
        if guidance.ndim != 4 or text.ndim != 3:
            raise ValueError("interaction expects guidance [B,D,H,W] and text [B,K,D]")
        if guidance.shape[0] != batch or guidance.shape[-2:] != (height, width) or text.shape[:2] != (batch, classes):
            raise ValueError("interaction guidance dimensions must match the cost volume")
        s = spatial.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        c = category.permute(0, 2, 1, 3, 4).reshape_as(s)
        feature = self.input_projection(torch.cat([s, c, (s - c).abs(), s * c], dim=1))
        visual = self.visual_projection(guidance)[:, None].expand(-1, classes, -1, -1, -1).reshape_as(s)
        semantic = self.text_projection(text).reshape(batch * classes, channels, 1, 1).expand_as(s)
        weights = self.router(torch.cat([feature, visual, semantic], dim=1)).float().softmax(dim=1).to(feature.dtype)
        mixture = sum(weights[:, i:i + 1] * expert(feature) for i, expert in enumerate(self.experts))
        output = 0.5 * (s + c) + self.scale * mixture
        output = output + self.ffn_scale * self.refine(output)
        return output.reshape(batch, classes, channels, height, width).permute(0, 2, 1, 3, 4)


class JointAffinityDecoder(nn.Module):
    """Per-class CAD feeds local semantic affinity reassembly at a 2x stage.

    A PyTorch reference operator inspired by SAPA's local affinity and FADE's
    joint conditioning. It is neither implementation. Existing transposed-conv
    cost is retained as an anchor. Out-of-bounds keys are masked before softmax.
    """

    def __init__(self, low_channels, cost_channels, guidance_channels, residual_scale=0.01,
                 kernel_size=3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("affinity kernel must be a positive odd number")
        self.kernel_size = kernel_size
        self.guidance_refiner = ClassGuidanceRefiner(cost_channels, guidance_channels, residual_scale)
        embed = max(guidance_channels, 8)
        self.query = nn.Sequential(
            group_norm(cost_channels + guidance_channels),
            nn.Conv2d(cost_channels + guidance_channels, embed, 1), nn.GELU(),
            nn.Conv2d(embed, embed, 1),
        )
        self.key = nn.Sequential(group_norm(low_channels), nn.Conv2d(low_channels, embed, 1))
        self.value = nn.Conv2d(low_channels, cost_channels, 1)
        self.scale = nn.Parameter(torch.full((1, cost_channels, 1, 1), float(residual_scale)))

    def reassemble(self, low, query, value):
        """Nearest-associated low-res neighborhood, high-res per-pixel weights."""
        batch, embed, height, width = low.shape
        size = query.shape[-2:]
        kernel = self.kernel_size
        neighbors = kernel ** 2
        keys = F.unfold(low, kernel, padding=kernel // 2).reshape(batch, embed * neighbors, height, width)
        keys = F.interpolate(keys, size=size, mode="nearest").reshape(batch, embed, neighbors, *size)
        logits = (query.float()[:, :, None] * keys.float()).sum(dim=1) / math.sqrt(embed)
        valid = F.unfold(low.new_ones(1, 1, height, width), kernel, padding=kernel // 2)
        valid = F.interpolate(valid.reshape(1, neighbors, height, width), size=size, mode="nearest")
        logits = logits.masked_fill(valid == 0, float("-inf"))
        weights = logits.softmax(dim=1).to(value.dtype)
        values = F.unfold(value, kernel, padding=kernel // 2).reshape(batch, value.shape[1], neighbors, height, width)
        # Avoid materializing all C * kernel^2 values at high resolution.
        result = 0
        for index in range(neighbors):
            shifted = F.interpolate(values[:, :, index], size=size, mode="nearest")
            result = result + shifted * weights[:, index:index + 1]
        return result

    def forward(self, low_cost, upsampled_cost, guidance):
        batch, classes = ClassGuidanceRefiner._validate(upsampled_cost, guidance)
        if low_cost.ndim != 4 or low_cost.shape[0] != batch * classes:
            raise ValueError("affinity low cost must retain the B*K ordering")
        if tuple(2 * d for d in low_cost.shape[-2:]) != upsampled_cost.shape[-2:]:
            raise ValueError("affinity decoder requires a 2x spatial stage")
        refined_guidance = self.guidance_refiner(upsampled_cost, guidance)
        query = self.query(torch.cat([upsampled_cost, refined_guidance], dim=1))
        dynamic = self.reassemble(self.key(low_cost), query, self.value(low_cost))
        output = upsampled_cost + self.scale * (dynamic - upsampled_cost)
        return output, refined_guidance
