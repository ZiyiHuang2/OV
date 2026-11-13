"""Rival-conditioned spatial transfer for CAT-Seg cost features.

An independent research prototype, not a validated performance improvement.
The class axis remains dynamic; no class IDs or learned per-class embeddings are used.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RivalConditionedTransfer(nn.Module):
    """Cost-conditioned local mixing with optional rival-feature messages.

    Args:
        dim: channel dimension of CAT-Seg's cost embedding.
        arm: S=spatial evidence only, M=margin-conditioned spatial,
             R=margin-conditioned spatial + rival-relative feature messages,
             P=pointwise rival-relative control (no spatial transfer).
    Inputs:
        x [B,D,K,H,W], scores [B,K,H,W] (pre-aggregation cosine scores),
        optional valid class [K]/[B,K], valid spatial [B,H,W] boolean masks.
    """

    def __init__(self, dim: int, arm: str = "R", *, rank: int = 24,
                 attn_dim: int = 8, init_scale: float = 0.01):
        super().__init__()
        if dim < 1 or rank < 1 or attn_dim < 1 or arm not in {"S", "M", "R", "P"}:
            raise ValueError("Invalid dim, rank, attn_dim, or arm")
        self.dim, self.arm, self.rank, self.attn_dim = dim, arm, rank, attn_dim
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Conv2d(dim, attn_dim, 1, bias=False)
        self.key = nn.Conv2d(dim, attn_dim, 1, bias=False)
        self.value = nn.Conv2d(dim, rank, 1, bias=False)
        self.pair = nn.Sequential(nn.Linear(7, 16), nn.GELU(), nn.Linear(16, 1))
        self.control = nn.Sequential(nn.Conv2d(4, 16, 1), nn.GELU(), nn.Conv2d(16, 2, 1))
        self.out = nn.Conv2d(rank, dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

    @staticmethod
    def _class_valid(scores, mask):
        b, k, _, _ = scores.shape
        if mask is None:
            return torch.ones((b, k), device=scores.device, dtype=torch.bool)
        if mask.dtype != torch.bool:
            raise ValueError("class_mask must be bool")
        if mask.shape == (k,):
            mask = mask[None].expand(b, -1)
        if mask.shape != (b, k) or not mask.any(1).all():
            raise ValueError("class_mask must be [K] or [B,K] with a valid class")
        return mask.to(scores.device)

    def _rival(self, scores, valid):
        b, k, h, w = scores.shape
        if k == 1:
            return torch.zeros((b, k, h, w), dtype=torch.long, device=scores.device), torch.zeros_like(scores), torch.zeros((b,), dtype=torch.bool, device=scores.device)
        ranked = scores.float().masked_fill(~valid[:, :, None, None], float("-inf"))
        top, idx = torch.topk(ranked, 2, dim=1)
        ids = torch.arange(k, device=scores.device)[None, :, None, None]
        winner = idx[:, :1] == ids
        other = torch.where(winner, idx[:, 1:2], idx[:, :1])
        enough = valid.sum(1) >= 2
        rival_score = scores.float().gather(1, other)
        margin = torch.where(enough[:, None, None, None], scores.float() - rival_score, torch.zeros_like(scores.float()))
        return other, margin, enough

    @staticmethod
    def _unfold_1(x):
        # x: [B*K, C, H, W] -> [B*K, N, 9, C]
        bk, c, h, w = x.shape
        return F.unfold(x, 3, padding=1).reshape(bk, c, 9, h*w).permute(0, 3, 2, 1)

    def forward(self, x, scores, class_mask=None, spatial_mask=None,
                *, enabled: bool = True, return_aux: bool = False):
        if not enabled:
            return (x, {"bypassed": True}) if return_aux else x
        if x.ndim != 5 or scores.ndim != 4:
            raise ValueError("Expected x[B,D,K,H,W], scores[B,K,H,W]")
        b, d, k, h, w = x.shape
        if d != self.dim or scores.shape != (b, k, h, w) or x.device != scores.device:
            raise ValueError("Mismatched dimensions or devices")
        if not x.is_floating_point() or not scores.is_floating_point() or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(scores).all()):
            raise ValueError("Features and scores must be finite floats")
        valid_class = self._class_valid(scores, class_mask)
        if spatial_mask is None:
            valid_space = torch.ones((b, h, w), device=x.device, dtype=torch.bool)
        elif spatial_mask.shape != (b, h, w) or spatial_mask.dtype != torch.bool:
            raise ValueError("spatial_mask must be bool [B,H,W]")
        else:
            valid_space = spatial_mask.to(x.device)

        opponent_id, margin, rival_present = self._rival(scores, valid_class)
        # Feature normalization is shared across categories and spatial locations.
        normalized = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        if self.arm in {"R", "P"}:
            opponent = normalized.gather(2, opponent_id[:, None].expand(-1, d, -1, -1, -1))
            relative = normalized - opponent
            relative = relative * rival_present[:, None, None, None, None]
        else:
            relative = normalized

        flatten = lambda z: z.permute(0, 2, 1, 3, 4).reshape(b*k, z.shape[1], h, w)
        flat = flatten(normalized)
        rel = flatten(relative)
        s = scores.float().reshape(b*k, 1, h, w)
        m = margin.reshape(b*k, 1, h, w) if self.arm != "S" else torch.zeros_like(s)
        # Local margin variation describes potential decision boundaries.
        local_mean = F.avg_pool2d(m, 3, stride=1, padding=1, count_include_pad=False)
        control_features = torch.cat([s, m, m.abs(), (m - local_mean).abs()], dim=1)
        controlled = self.control(control_features)
        gate = F.hardsigmoid(controlled[:, 0:1])
        mix = torch.sigmoid(controlled[:, 1:2])

        if self.arm == "P":
            delta = self.value(rel)
        else:
            representation = rel if self.arm == "R" else flat
            q = self.query(representation).flatten(2).transpose(1, 2)
            neighbors_k = self._unfold_1(self.key(representation))
            dot = (q[:, :, None, :] * neighbors_k).sum(-1) / math.sqrt(self.attn_dim)

            sj = self._unfold_1(s).squeeze(-1)
            mj = self._unfold_1(m).squeeze(-1)
            si = s.flatten(2).transpose(1, 2).expand_as(sj)
            mi = m.flatten(2).transpose(1, 2).expand_as(mj)
            if self.arm == "R":
                opponent = opponent_id.reshape(b*k, 1, h, w).float()
                rival_neighbors = self._unfold_1(opponent).squeeze(-1)
                same_rival = (rival_neighbors == opponent.flatten(2).transpose(1, 2)).float()
            else:
                same_rival = torch.zeros_like(sj)
            relation = torch.stack([si, sj, sj-si, mi, mj, (mj-mi).abs(), same_rival], dim=-1)
            pair_logits = self.pair(relation.to(self.pair[0].weight.dtype)).squeeze(-1).float()

            vspace = valid_space[:, None].expand(b, k, h, w).reshape(b*k, 1, h, w)
            neighbor_ok = self._unfold_1(vspace.float()).squeeze(-1).bool()
            neighbor_ok[:, :, 4] = False
            neighbor_ok &= vspace.flatten(2).transpose(1, 2).bool()
            neighbor_ok &= valid_class.reshape(b*k, 1, 1)
            weights = torch.softmax((pair_logits + dot.float()).masked_fill(~neighbor_ok, -1e4), dim=-1)
            weights = weights * neighbor_ok.float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)

            def local_delta(tensor):
                value = self.value(tensor)
                neighbors = self._unfold_1(value)
                center = value.flatten(2).transpose(1, 2)
                result = ((neighbors-center[:, :, None]) * weights[:, :, :, None].to(neighbors.dtype)).sum(2)
                return result.transpose(1, 2).reshape(b*k, self.rank, h, w)

            spatial_delta = local_delta(flat)
            if self.arm == "R":
                relative_delta = local_delta(rel)
                delta = (1-mix.to(spatial_delta.dtype))*spatial_delta + mix.to(spatial_delta.dtype)*relative_delta
            else:
                delta = spatial_delta

        update = self.out(delta).to(x.dtype) * gate.to(x.dtype) * self.alpha.to(x.dtype)
        active = (valid_class[:, :, None, None] & valid_space[:, None]).reshape(b*k, 1, h, w)
        if self.arm in {"R", "P"}:
            active &= rival_present[:, None, None, None].expand(b, k, h, w).reshape(b*k, 1, h, w)
        update = update * active.to(x.dtype)
        result = x + update.reshape(b, k, d, h, w).permute(0, 2, 1, 3, 4)
        if return_aux:
            return result, {"bypassed": False, "gate_mean": gate.detach().mean(),
                            "mix_mean": mix.detach().mean(),
                            "update_rms": update.detach().float().square().mean().sqrt(),
                            "alpha": self.alpha.detach()}
        return result
