"""M1: LSK-inspired shared visual guidance and its convolution control."""
import torch
from torch import nn


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
