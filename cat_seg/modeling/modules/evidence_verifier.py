"""Optional relation-aware evidence verification for OV cost maps."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CounterfactualEvidenceVerifier(nn.Module):
    """Compare each class response with text-related competitors.

    The module is identity-safe: its learnable residual scale is initialized
    from ``residual_scale`` (zero by default), so enabling the module does not
    silently change the baseline before a controlled pilot is requested.
    ``correlation`` has shape ``[B, P, T, H, W]`` and ``text_features`` has
    shape ``[B, T, P, C]``.
    """

    def __init__(self, temperature=0.07, margin_temperature=0.10,
                 residual_scale=0.0, topk=0):
        super().__init__()
        if temperature <= 0 or margin_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if topk < 0:
            raise ValueError("topk must be non-negative")
        self.temperature = float(temperature)
        self.margin_temperature = float(margin_temperature)
        self.topk = int(topk)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    @staticmethod
    def _relation_weights(text_features, temperature, topk):
        if text_features.ndim != 4:
            raise ValueError("text_features must have shape [B, T, P, C]")
        text = F.normalize(text_features.mean(dim=2), dim=-1)
        relation = torch.einsum("btc,bdc->btd", text, text)
        classes = relation.shape[-1]
        if classes == 1:
            return torch.zeros_like(relation)
        diagonal = torch.eye(classes, device=relation.device,
                             dtype=torch.bool).unsqueeze(0)
        relation = relation.masked_fill(diagonal, float("-inf"))
        if 0 < topk < classes:
            values, indices = relation.topk(topk, dim=-1)
            sparse = torch.full_like(relation, float("-inf"))
            relation = sparse.scatter(-1, indices, values)
        return F.softmax(relation / temperature, dim=-1)

    def forward(self, correlation, text_features) -> Tuple[torch.Tensor, dict]:
        if correlation.ndim != 5:
            raise ValueError("correlation must have shape [B, P, T, H, W]")
        if text_features.ndim != 4:
            raise ValueError("text_features must have shape [B, T, P, C]")
        if (correlation.shape[0] != text_features.shape[0]
                or correlation.shape[2] != text_features.shape[1]):
            raise ValueError("correlation and text_features have incompatible dimensions")

        positive = correlation.mean(dim=1)
        weights = self._relation_weights(text_features, self.temperature, self.topk)
        negative = torch.einsum("btd,bdhw->bthw", weights, positive)
        margin = positive - negative
        has_competitor = (weights.sum(dim=-1) > 0).unsqueeze(-1).unsqueeze(-1)
        margin = margin * has_competitor
        gate = torch.sigmoid(margin / self.margin_temperature)
        residual = self.residual_scale * gate * torch.tanh(margin)
        verified = correlation + residual.unsqueeze(1)
        diagnostics = {
            "positive": positive.detach(),
            "negative": negative.detach(),
            "margin": margin.detach(),
            "gate": gate.detach(),
            "relation": weights.detach(),
        }
        return verified, diagnostics
