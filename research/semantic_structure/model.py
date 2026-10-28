"""Version 2: controlled feature-decoder interventions, not an HG reproduction."""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_VERSION = 2

class SemanticStructureDecoder(nn.Module):
    def __init__(self, clip_dim, structure_dims, width=64, semantic_mode='mean',
                 spatial_mode='geometry', temperature=0.1, semantic_strength=1.0):
        super().__init__()
        if semantic_mode not in {'mean', 'adaptive'}:
            raise ValueError('semantic_mode: mean or adaptive')
        if spatial_mode not in {'identity', 'geometry', 'conditioned'}:
            raise ValueError('spatial_mode: identity, geometry or conditioned; v1 names are unsupported')
        if not structure_dims or min(clip_dim, width, *structure_dims) < 1:
            raise ValueError('Positive dimensions and a nonempty structure pyramid required')
        if not 0 < temperature < float('inf') or not 0 <= semantic_strength < float('inf'):
            raise ValueError('Invalid fixed intervention coefficients')
        self.config = dict(clip_dim=clip_dim, structure_dims=list(structure_dims), width=width,
                           semantic_mode=semantic_mode, spatial_mode=spatial_mode,
                           temperature=temperature, semantic_strength=semantic_strength)
        self.semantic_mode, self.spatial_mode = semantic_mode, spatial_mode
        self.temperature, self.semantic_strength = temperature, semantic_strength
        # All arms have exactly the same learnable parameters and initialization order.
        self.evidence = nn.Conv2d(clip_dim, width, 1)
        self.structure = nn.ModuleList(nn.Conv2d(d, width, 1) for d in structure_dims)
        self.refine = nn.ModuleList(nn.Sequential(nn.Conv2d(4*width, width, 3, padding=1),
            nn.GELU(), nn.Conv2d(width, width, 3, padding=1)) for _ in structure_dims)
        self.head = nn.Conv2d(width, 1, 1)

    def selected_text(self, visual, descriptors):
        if self.semantic_mode == 'mean':
            selected = descriptors.mean(0)[None, :, None, None]
        else:
            scores = torch.einsum('bdhw,pd->bphw', visual, descriptors)
            weights = (scores / self.temperature).softmax(1)
            selected = torch.einsum('bphw,pd->bdhw', weights, descriptors)
        # Identical normalization removes the former magnitude confound.
        return F.normalize(selected, dim=1)

    def build_evidence(self, visual, text, valid):
        if visual.ndim != 4 or text.ndim != 3 or valid.shape != text.shape[:2]:
            raise ValueError('Expected visual BCHW, text KPD, valid KP')
        if visual.shape[1] != text.shape[-1] or valid.dtype != torch.bool or len(text) == 0:
            raise ValueError('Invalid aligned dimensions, mask or empty vocabulary')
        if not valid.any(-1).all():
            raise ValueError('Each class needs at least one valid description')
        if not torch.isfinite(visual).all() or not torch.isfinite(text[valid]).all():
            raise ValueError('Nonfinite valid input')
        v = F.normalize(visual.float(), dim=1)
        outputs = []
        for descriptions, mask in zip(text, valid):
            t = F.normalize(descriptions[mask].float(), dim=-1)
            selected = self.selected_text(v, t)
            outputs.append(self.evidence((v * selected).to(visual.dtype)))
        return torch.stack(outputs, dim=1)

    def transport(self, h, geometry, evidence):
        if self.spatial_mode == 'identity':
            return h
        b, k, d, height, width = h.shape
        n = height * width
        g = F.normalize(geometry.float(), dim=1)
        gn = F.unfold(g, 3, padding=1).reshape(b, d, 9, n)
        affinity = (g.flatten(2).unsqueeze(2) * gn).sum(1)
        boundary = F.unfold(torch.ones_like(g[:, :1]), 3, padding=1).bool()
        result = []
        for c in range(k):
            scores = affinity
            if self.spatial_mode == 'conditioned':
                e = F.normalize(evidence[:, c].float(), dim=1)
                en = F.unfold(e, 3, padding=1).reshape(b, d, 9, n)
                semantic = (e.flatten(2).unsqueeze(2) * en).sum(1)
                scores = scores + self.semantic_strength * semantic
            weights = scores.masked_fill(~boundary, float('-inf')).softmax(1)
            neighbors = F.unfold(h[:, c], 3, padding=1).reshape(b, d, 9, n)
            result.append((neighbors * weights[:, None].to(h.dtype)).sum(2)
                          .reshape(b, d, height, width))
        return torch.stack(result, dim=1)

    def forward(self, visual, text, valid, structures, output_size=None):
        if len(structures) != len(self.structure):
            raise ValueError('Wrong number of structure levels')
        evidence = self.build_evidence(visual, text, valid)
        b, k, d, _, _ = evidence.shape
        state = evidence.flatten(0, 1)
        previous_size = (0, 0)
        for features, projection, refine in zip(structures, self.structure, self.refine):
            if features.ndim != 4 or features.shape[0] != b or not torch.isfinite(features).all():
                raise ValueError('Invalid structure features')
            size = features.shape[-2:]
            if any(a < z for a, z in zip(size, previous_size)):
                raise ValueError('Structure levels must be coarse-to-fine')
            previous_size = size
            state = F.interpolate(state, size=size, mode='bilinear', align_corners=False)
            direct = F.interpolate(evidence.flatten(0, 1), size=size, mode='bilinear', align_corners=False)
            geometry = projection(features)
            guided = self.transport(state.reshape(b, k, d, *size), geometry,
                                    direct.reshape(b, k, d, *size)).flatten(0, 1)
            # Keep the SAME direct structure and evidence paths in all arms.
            shared = geometry[:, None].expand(-1, k, -1, -1, -1).reshape(b*k, d, *size)
            state = state + refine(torch.cat((state, direct, shared, guided), dim=1))
        logits = self.head(state).reshape(b, k, *previous_size)
        if output_size is not None:
            logits = F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)
        return logits


def segmentation_loss(logits, labels, ignore_index=255):
    if not (labels != ignore_index).any():
        return logits.sum() * 0
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def optimization_step(model, optimizer, visual, text, valid, structures, labels):
    # Even zero gradients can cause AdamW decay/momentum updates: skip the entire step.
    optimizer.zero_grad(set_to_none=True)
    if not (labels != 255).any():
        return None
    logits = model(visual, text, valid, structures, labels.shape[-2:])
    loss = segmentation_loss(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())
