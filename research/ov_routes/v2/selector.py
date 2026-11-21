"""Spatial-semantic, action-comparative selector (GT-free at inference).

A spatial encoder models *where* first-pass errors may occur; an optional
cross-attention injects top-ranked CLIP class semantics into spatial tokens;
optional cross-action attention compares eight windows before gain regression.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class SpatialSemanticSelector(nn.Module):
    """Estimate *scaled BCE benefit* of every crop; larger = better.

    All inputs are batched by image: spatial [B,A,8,G,G], legacy [B,A,D+9],
    class_tokens [B,A,M,D+2], class_valid [B,A,M].

    Ablations: ``use_spatial`` removes the spatial/cross-text branch,
    ``use_cross`` toggles text->spatial attention (only if spatial enabled),
    ``use_action_attention`` toggles cross-window comparison.
    """

    def __init__(
        self,
        text_dim: int,
        grid_size: int = 8,
        hidden_dim: int = 64,
        top_m: int = 4,
        use_spatial: bool = True,
        use_cross: bool = True,
        use_action_attention: bool = True,
        use_raw_cost: bool = False,
    ):
        super().__init__()
        if hidden_dim % 8 or grid_size < 2 or top_m < 2 or text_dim < 2:
            raise ValueError('Invalid hidden_dim, grid_size, top_m or text_dim')
        if use_cross and not use_spatial:
            raise ValueError('Cross attention needs a spatial branch')
        self.text_dim, self.grid_size, self.top_m = text_dim, grid_size, top_m
        self.use_spatial, self.use_cross = use_spatial, use_cross
        self.use_action_attention = use_action_attention
        self.use_raw_cost = use_raw_cost

        self.legacy = nn.Sequential(
            nn.Linear(text_dim+9, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        if use_spatial:
            self.spatial = nn.Sequential(
                nn.Conv2d(12 if use_raw_cost else 8, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim), nn.GELU(),
                ResidualSpatialBlock(hidden_dim),
            )
            self.spatial_norm = nn.LayerNorm(hidden_dim)
            self.position = nn.Parameter(torch.zeros(1, grid_size*grid_size, hidden_dim))
            nn.init.normal_(self.position, std=.02)
            if use_cross:
                self.text_projection = nn.Sequential(
                    nn.Linear(text_dim+2, hidden_dim), nn.LayerNorm(hidden_dim),
                )
                self.cross = nn.MultiheadAttention(hidden_dim, 4, dropout=0, batch_first=True)
                self.cross_norm = nn.LayerNorm(hidden_dim)
            self.spatial_pool = nn.Linear(hidden_dim, 1)
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        if use_action_attention:
            self.action_norm = nn.LayerNorm(hidden_dim)
            self.action_attention = nn.MultiheadAttention(hidden_dim, 4, dropout=0, batch_first=True)
            self.action_postnorm = nn.LayerNorm(hidden_dim)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(),
            nn.Linear(hidden_dim//2, 1),
        )

    def forward(self, evidence: dict[str, torch.Tensor]) -> torch.Tensor:
        p = evidence['legacy']
        maps = evidence['spatial']
        tokens = evidence['class_tokens']
        mask = evidence['class_valid']
        if p.ndim != 3 or maps.ndim != 5 or tokens.ndim != 4 or mask.ndim != 3:
            raise ValueError('Provide batched evidence: [B,A,...]')
        b, a, legacy_dim = p.shape
        if legacy_dim != self.text_dim+9 or maps.shape[:2] != (b,a) or maps.shape[-2:] != (self.grid_size,self.grid_size):
            raise ValueError('Legacy or spatial dimension mismatch')
        channels = 12 if self.use_raw_cost else 8
        if maps.shape[2] < channels:
            raise ValueError('The selector requests raw cost maps absent from input')
        if tokens.shape != (b,a,self.top_m,self.text_dim+2) or mask.shape != (b,a,self.top_m):
            raise ValueError('Text token or mask dimension mismatch')
        if not mask.any(-1).all():
            raise ValueError('Each action needs at least one real class token')

        h = self.legacy(p)
        if self.use_spatial:
            z = self.spatial(maps[:,:,:channels].reshape(b*a,channels,self.grid_size,self.grid_size))
            z = z.flatten(2).transpose(1,2)
            z = self.spatial_norm(z+self.position)
            if self.use_cross:
                t = self.text_projection(tokens.reshape(b*a,self.top_m,self.text_dim+2))
                attended, _ = self.cross(z,t,t,key_padding_mask=~mask.reshape(b*a,self.top_m),need_weights=False)
                z = self.cross_norm(z+attended)
            weights = self.spatial_pool(z).softmax(1)
            z = (z*weights).sum(1).reshape(b,a,-1)
            h = self.fuse(torch.cat((h,z), dim=-1))

        if self.use_action_attention:
            z, _ = self.action_attention(
                self.action_norm(h),self.action_norm(h),self.action_norm(h),
                need_weights=False,
            )
            h = self.action_postnorm(h+z)
        return self.gain_head(h).squeeze(-1)


class LegacySelectorExact(nn.Module):
    """Exact Route B V1 selector *architecture* for matched-cache controls.

    It can load the original ViewSelector state_dict (`net.0.*`, `net.2.*`)
    when hidden_dim=64. Training augmentation/optimizer are controlled by the
    new runner, and thus differ from the historical V1 training run.
    """
    use_raw_cost = False

    def __init__(self, text_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(text_dim+9,hidden_dim),
                                 nn.GELU(),nn.Linear(hidden_dim,1))

    def forward(self, evidence: dict[str,torch.Tensor]) -> torch.Tensor:
        x=evidence['legacy']
        if x.ndim!=3:
            raise ValueError('Expected [B,A,D+9] legacy evidence')
        return self.net(x).squeeze(-1)


class HeuristicSelector(nn.Module):
    """Parameter-free initial-score controls, using the exact V1 summary slots.

    These are not *calibrated gains*. They only rank the eight fixed actions.
    All choices are GT-free and use one extra crop encoding, like learned V2.
    """
    use_raw_cost = False

    def __init__(self, policy: str):
        super().__init__()
        if policy not in ('entropy', 'low_margin', 'disagreement'):
            raise ValueError(f'Unsupported heuristic: {policy}')
        self.policy = policy

    def forward(self, evidence: dict[str,torch.Tensor]):
        features = evidence['legacy']
        if features.ndim != 3 or features.shape[-1] < 10:
            raise ValueError('Heuristic expects batched V1 features')
        dim = features.shape[-1] - 9
        if self.policy == 'entropy':
            return features[...,dim+3]
        if self.policy == 'low_margin':
            return -features[...,dim+2]
        return features[...,dim+4]


class IndexSelector(nn.Module):
    """Choose a predetermined one of A windows without GT and without learning."""
    use_raw_cost = False

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, evidence):
        n, actions = evidence['legacy'].shape[:2]
        if not 0 <= self.index < actions:
            raise ValueError('Window index outside action set')
        scores = evidence['legacy'].new_zeros(n, actions)
        scores[:,self.index] = 1.
        return scores
