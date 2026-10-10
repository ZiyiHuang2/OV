"""Normalization shared by the aggregation and decoder adaptations."""
from torch import nn


def group_norm(channels):
    groups = min(32, max(channels // 4, 1))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)
