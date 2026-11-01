"""Minimal B/C conditional spatial *residual* unit for CAT-Seg.

Independent of Detectron2 so that math and invariants can be tested on CPU.
Inputs:
    x: [B,D,K,H,W], features AFTER one native Swin spatial stage.
    scores: [B,K,H,W], raw, prompt-reduced CLIP correlation (not decoder logits).
The optional extra path never replaces native Swin or class aggregation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CompetitiveSpatialUpdate(nn.Module):
    """B=self-only local contrast; C=cross-class competition contrast.

    The *sole* B/C implementation difference is the reference against which
    each current-class score is compared. All learned layers and feature
    dimensions are identical. The contrast is not a calibrated confidence.
    """

    def __init__(self, dim: int, mode: str, hidden: int = 24,
                 initial_scale: float = 0.01, window: int = 3):
        super().__init__()
        if dim < 1 or hidden < 1 or window != 3:
            raise ValueError("dim/hidden must be positive; v1 supports a 3x3 window")
        if mode not in ("self_only", "competitive"):
            raise ValueError("mode must be self_only (B) or competitive (C)")
        self.dim = dim
        self.mode = mode
        self.window = window
        self.relation = nn.Sequential(nn.Linear(8, hidden), nn.GELU(),
                                      nn.Linear(hidden, 1, bias=False))
        self.accept = nn.Sequential(nn.Linear(2, hidden), nn.GELU(),
                                    nn.Linear(hidden, 1))
        self.value = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.value.weight)
        self.alpha = nn.Parameter(torch.tensor(float(initial_scale)))

    @staticmethod
    def _valid_class_mask(scores: torch.Tensor, class_mask: torch.Tensor | None):
        b, k, _, _ = scores.shape
        if class_mask is None:
            return torch.ones((b, k), device=scores.device, dtype=torch.bool)
        if class_mask.ndim == 1 and class_mask.shape[0] == k:
            class_mask = class_mask.unsqueeze(0).expand(b, -1)
        if class_mask.shape != (b, k) or class_mask.dtype != torch.bool:
            raise ValueError("class_mask must be bool [K] or [B,K]")
        if not class_mask.any(dim=1).all():
            raise ValueError("Each image needs at least one valid class")
        return class_mask.to(scores.device)

    @staticmethod
    def _self_local_reference(scores: torch.Tensor, spatial_valid: torch.Tensor):
        """Neighbor-only mean of SAME class (no cross-class information)."""
        b, k, h, w = scores.shape
        flat = scores.float().reshape(b*k, 1, h, w)
        valid = spatial_valid[:, None].expand(b, k, h, w).reshape(b*k, 1, h, w)
        kernel = torch.ones((1, 1, 3, 3), dtype=flat.dtype, device=flat.device)
        kernel[0, 0, 1, 1] = 0.
        total = F.conv2d(flat * valid, kernel, padding=1)
        count = F.conv2d(valid, kernel, padding=1)
        mean = torch.where(count > 0, total / count.clamp_min(1), flat)
        return mean.reshape(b, k, h, w)

    def contrast(self, scores: torch.Tensor,
                 class_mask: torch.Tensor | None = None,
                 spatial_mask: torch.Tensor | None = None):
        """B and C use the same contrast formula with different references.

        B: s_c - mean of neighboring s_c.
        C: s_c - max(s_other_classes); for a single valid class falls back to B.
        Class count and vocabulary are dynamic. The max excludes padded classes.
        """
        if scores.ndim != 4 or scores.shape[1] < 1 or not torch.isfinite(scores).all():
            raise ValueError("scores must be finite [B,K,H,W]")
        b, k, h, w = scores.shape
        valid_cls = self._valid_class_mask(scores, class_mask)
        if spatial_mask is None:
            valid_space = torch.ones((b, h, w), dtype=torch.bool, device=scores.device)
        else:
            if spatial_mask.shape != (b, h, w) or spatial_mask.dtype != torch.bool:
                raise ValueError("spatial_mask must be bool [B,H,W]")
            valid_space = spatial_mask.to(scores.device)
        s = scores.float()
        self_ref = self._self_local_reference(s, valid_space.float())
        if self.mode == "self_only" or k == 1:
            ref = self_ref
        else:
            masked = s.masked_fill(~valid_cls[:, :, None, None], float("-inf"))
            top, idx = torch.topk(masked, k=2, dim=1)
            is_top = torch.zeros_like(masked, dtype=torch.bool).scatter_(1, idx[:, :1], True)
            competitor = torch.where(is_top, top[:, 1:2], top[:, :1])
            ref = torch.where((valid_cls.sum(1) > 1)[:, None, None, None],
                              competitor, self_ref)
        t = s - ref
        return torch.where(valid_cls[:, :, None, None], t, torch.zeros_like(t))

    def forward(self, x: torch.Tensor, scores: torch.Tensor,
                class_mask: torch.Tensor | None = None,
                spatial_mask: torch.Tensor | None = None,
                *, enabled: bool = True, return_aux: bool = False):
        if not enabled:
            return (x, {"bypassed": True}) if return_aux else x
        if x.ndim != 5 or scores.ndim != 4:
            raise ValueError("Expected x[B,D,K,H,W], scores[B,K,H,W]")
        b, d, k, h, w = x.shape
        if d != self.dim or scores.shape != (b, k, h, w) or not torch.isfinite(x).all():
            raise ValueError("Invalid feature/score shapes or nonfinite inputs")
        valid_cls = self._valid_class_mask(scores, class_mask)
        if spatial_mask is None:
            valid_space = torch.ones((b, h, w), device=x.device, dtype=torch.bool)
        else:
            if spatial_mask.shape != (b, h, w) or spatial_mask.dtype != torch.bool:
                raise ValueError("spatial_mask must be bool [B,H,W]")
            valid_space = spatial_mask.to(x.device)
        if scores.device != x.device:
            raise ValueError("scores and features must share a device")
        cond = self.contrast(scores, valid_cls, valid_space)
        n = h*w
        q = 9
        # [B,K,N,D] centers and [B,K,N,9,D] neighbors.
        flat = x.permute(0, 2, 1, 3, 4).reshape(b*k, d, h, w)
        center = flat.flatten(2).permute(0, 2, 1).reshape(b, k, n, d)
        neigh = F.unfold(flat, kernel_size=3, padding=1).reshape(
            b, k, d, q, n).permute(0, 1, 4, 3, 2)
        scores_f = scores.float().reshape(b*k, 1, h, w)
        cond_f = cond.reshape(b*k, 1, h, w)
        sj = F.unfold(scores_f, 3, padding=1).reshape(b, k, q, n).permute(0, 1, 3, 2)
        tj = F.unfold(cond_f, 3, padding=1).reshape(b, k, q, n).permute(0, 1, 3, 2)
        si = scores.float().reshape(b, k, n).unsqueeze(-1).expand_as(sj)
        ti = cond.reshape(b, k, n).unsqueeze(-1).expand_as(tj)
        # One directional pairwise function for both modes. Receivers and
        # neighbors interact nonlinearly: not receiver-only softmax constants.
        features = torch.stack([si, sj, sj-si, si*sj, ti, tj,
                                ti*tj, (ti-tj).abs()], dim=-1)
        pair_logits = self.relation(features.to(self.relation[0].weight.dtype)).squeeze(-1).float()
        space_neighbors = F.unfold(valid_space.float().unsqueeze(1),
                                   kernel_size=3, padding=1).permute(0, 2, 1).bool()
        space_neighbors = space_neighbors.clone()
        space_neighbors[:, :, 4] = False  # Exclude self from additional message.
        allowed = (space_neighbors[:, None] &
                   valid_space.reshape(b, 1, n, 1) &
                   valid_cls[:, :, None, None])
        # All-invalid rows yield zero updates, not NaNs or a forced neighbor.
        weights = F.softmax(pair_logits.masked_fill(~allowed, -1e4), dim=-1)
        weights = weights * allowed.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        delta = ((neigh - center.unsqueeze(-2)) * weights[..., None].to(x.dtype)).sum(-2)
        delta = self.value(delta.to(self.value.weight.dtype)).to(x.dtype)
        # Accept controls UPDATE MAGNITUDE separately from neighbor selection.
        # Hard sigmoid can reach EXACTLY zero at an individual position;
        # ordinary sigmoid can only approach rejection asymptotically.
        gate_input = torch.stack([
            scores.float().reshape(b, k, n), cond.reshape(b, k, n)
        ], dim=-1).to(self.accept[0].weight.dtype)
        gate = F.hardsigmoid(self.accept(gate_input)).float()
        updated = center + self.alpha.to(x.dtype) * gate.to(x.dtype) * delta
        updated = torch.where(
            (valid_cls[:, :, None] & valid_space.reshape(b, 1, n))[..., None],
            updated, center)
        y = updated.reshape(b, k, h, w, d).permute(0, 4, 1, 2, 3).contiguous()
        if return_aux:
            return y, {"contrast": cond.detach(), "gate": gate.detach(),
                       "pair_logits": pair_logits.detach(), "allowed": allowed,
                       "alpha": self.alpha.detach(), "bypassed": False}
        return y
