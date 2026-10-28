"""Experimental feature decoder; not an official reproduction of the three papers."""
import torch
from torch import nn
from torch.nn import functional as F

class SemanticStructureDecoder(nn.Module):
    """Inputs: aligned CLIP BCHW, text KPD, valid KP, structure list BCHW.
    Structure pyramid must be ordered coarse-to-fine. Encoders are external.
    Parameters are shared across classes; no learned class-ID embeddings.
    """
    def __init__(self, clip_dim, structure_dims, width=64,
                 semantic_mode="adaptive", spatial_mode="transport"):
        super().__init__()
        if semantic_mode not in {"mean", "adaptive"}:
            raise ValueError("semantic_mode must be mean or adaptive")
        if spatial_mode not in {"fusion", "transport"}:
            raise ValueError("spatial_mode must be fusion or transport")
        if not structure_dims:
            raise ValueError("structure pyramid must not be empty")
        self.semantic_mode, self.spatial_mode = semantic_mode, spatial_mode
        self.evidence = nn.Conv2d(clip_dim, width, 1)
        self.structure = nn.ModuleList(nn.Conv2d(d, width, 1) for d in structure_dims)
        self.refine = nn.ModuleList(nn.Sequential(
            nn.Conv2d(3 * width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1)) for _ in structure_dims)
        self.head = nn.Conv2d(width, 1, 1)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.semantic_strength = nn.Parameter(torch.tensor(0.0))

    def build_evidence(self, visual, text, valid):
        if visual.ndim != 4 or text.ndim != 3 or valid.shape != text.shape[:2]:
            raise ValueError("Expected BCHW, KPD, KP")
        if visual.shape[1] != text.shape[-1] or valid.dtype != torch.bool:
            raise ValueError("Aligned dimensions and boolean validity mask required")
        if not valid.any(-1).all():
            raise ValueError("Each class needs at least one valid description")
        if not torch.isfinite(visual).all() or not torch.isfinite(text).all():
            raise ValueError("Nonfinite input")
        v = F.normalize(visual.float(), dim=1)
        t = F.normalize(text.float(), dim=-1)
        outputs = []
        # Class loop avoids materializing B*K*P*D*H*W.
        for descriptions, mask in zip(t, valid):
            descriptors = descriptions[mask]
            if self.semantic_mode == "mean":
                selected = F.normalize(descriptors.mean(0), dim=0)[None, :, None, None]
            else:
                scores = torch.einsum("bdhw,pd->bphw", v, descriptors)
                temperature = self.log_temperature.exp().clamp(0.05, 20)
                weights = (scores / temperature).softmax(1)
                selected = torch.einsum("bphw,pd->bdhw", weights, descriptors)
            outputs.append(self.evidence((v * selected).to(visual.dtype)))
        return torch.stack(outputs, dim=1)

    def transport(self, h, geometry, evidence):
        b, k, d, height, width = h.shape
        n = height * width
        g = F.normalize(geometry.float(), dim=1)
        gn = F.unfold(g, 3, padding=1).reshape(b, d, 9, n)
        affinity = (g.flatten(2).unsqueeze(2) * gn).sum(1)
        boundary = F.unfold(torch.ones_like(g[:, :1]), 3, padding=1).bool()
        result = []
        for c in range(k):
            e = F.normalize(evidence[:, c].float(), dim=1)
            en = F.unfold(e, 3, padding=1).reshape(b, d, 9, n)
            semantic = (e.flatten(2).unsqueeze(2) * en).sum(1)
            scores = affinity + self.semantic_strength.sigmoid() * semantic
            weights = scores.masked_fill(~boundary, float("-inf")).softmax(1)
            neighbors = F.unfold(h[:, c], 3, padding=1).reshape(b, d, 9, n)
            result.append((neighbors * weights[:, None].to(h.dtype)).sum(2)
                          .reshape(b, d, height, width))
        return torch.stack(result, dim=1)

    def forward(self, visual, text, valid, structures, output_size=None):
        if len(structures) != len(self.structure):
            raise ValueError("Wrong number of structure levels")
        evidence = self.build_evidence(visual, text, valid)
        b, k, d, _, _ = evidence.shape
        state = evidence.flatten(0, 1)
        previous_size = (0, 0)
        for features, projection, refine in zip(structures, self.structure, self.refine):
            if features.ndim != 4 or features.shape[0] != b:
                raise ValueError("Invalid structure batch")
            size = features.shape[-2:]
            if any(a < z for a, z in zip(size, previous_size)):
                raise ValueError("Structure levels must be coarse-to-fine")
            previous_size = size
            state = F.interpolate(state, size=size, mode="bilinear", align_corners=False)
            direct = F.interpolate(evidence.flatten(0, 1), size=size,
                                   mode="bilinear", align_corners=False)
            geometry = projection(features)
            if self.spatial_mode == "transport":
                guided = self.transport(state.reshape(b, k, d, *size), geometry,
                                        direct.reshape(b, k, d, *size)).flatten(0, 1)
            else:
                guided = geometry[:, None].expand(-1, k, -1, -1, -1).reshape(b*k, d, *size)
            state = state + refine(torch.cat((state, direct, guided), dim=1))
        logits = self.head(state).reshape(b, k, *previous_size)
        if output_size is not None:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

def segmentation_loss(logits, labels, ignore_index=255):
    """All-ignore batches have a finite zero loss and zero direct gradient."""
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)
