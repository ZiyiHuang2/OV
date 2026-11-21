"""GT-free, vocabulary-order-equivariant evidence for eight fixed crop actions.

This uses the already available CAT-Seg first-pass logits and CLIP text
embeddings.  No crop is re-encoded while constructing selector features.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
from torch.nn import functional as F

def windows(height: int, width: int):
    """Exact eight-action geometry of Route B v1 (two scales x four corners)."""
    result = []
    for scale in (.5, .75):
        h, w = max(1, round(height*scale)), max(1, round(width*scale))
        for y in (0, height-h):
            for x in (0, width-w):
                box = (y, x, y+h, x+w)
                if box not in result:
                    result.append(box)
    return result


Box = Tuple[int, int, int, int]


def build_evidence(
    logits: torch.Tensor,
    text_vectors: torch.Tensor,
    boxes: Sequence[Box] | None = None,
    *,
    grid_size: int = 8,
    top_m: int = 4,
    raw_cost: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Create eight per-action descriptors using *only* inference-time data.

    Args:
        logits: K,H,W float logits from host.predict_logits (not GT).
        text_vectors: K,D CLIP text embeddings in the SAME class order.
        boxes: coordinates on the original logits/image grid.
    Returns:
        spatial [A,8 or 12,G,G]: per-crop maps; optional 4 first-pass raw cost maps.
        legacy [A,D+9]: exact first-version text-difference/scalar features.
        class_tokens [A,M,D+2]: top-M ordered text embeddings plus local
            mean probability and winner occupancy; padded with zeros.
        class_valid [A,M]: True on real class tokens, False on pads.
    All tensors are detached. No label or GT argument exists.
    """
    if logits.ndim != 3 or text_vectors.ndim != 2:
        raise ValueError('Expected logits [K,H,W] and text [K,D]')
    k, height, width = logits.shape
    if k < 2 or text_vectors.shape[0] != k or grid_size < 2 or top_m < 2:
        raise ValueError('Invalid class count, text alignment, grid or top_m')
    if not torch.isfinite(logits).all() or not torch.isfinite(text_vectors).all():
        raise ValueError('Nonfinite logits/text')
    if raw_cost is not None:
        if raw_cost.ndim != 3 or raw_cost.shape[0] != k or not torch.isfinite(raw_cost).all():
            raise ValueError('raw_cost must be finite [K,h,w] from the same first pass')
    boxes = tuple(windows(height, width) if boxes is None else boxes)
    if not boxes:
        raise ValueError('At least one candidate window required')

    with torch.no_grad():
        prob = logits.float().softmax(dim=0)
        text = F.normalize(text_vectors.float(), dim=-1)
        best2, winner2 = prob.topk(2, dim=0)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(0) / math.log(k)
        top1, top2 = best2[0], best2[1]
        margin = top1 - top2
        win_ids = winner2[0]
        boundary_y = F.pad((win_ids[1:] != win_ids[:-1]).float(), (0, 0, 1, 0))
        boundary_x = F.pad((win_ids[:, 1:] != win_ids[:, :-1]).float(), (1, 0, 0, 0))
        boundary = (boundary_y + boundary_x) * .5
        dy = F.pad((top1[1:] - top1[:-1]).abs(), (0, 0, 1, 0))
        dx = F.pad((top1[:, 1:] - top1[:, :-1]).abs(), (1, 0, 0, 0))
        gradient = .5 * (dx + dy)

        # Upsampling here aligns the 24x24 CLIP cost map to image crop coordinates.
        # This uses pre-aggregation CLIP evidence, not GT or a second encoding.
        raw = None if raw_cost is None else F.interpolate(
            raw_cost.float().unsqueeze(0),size=(height,width),
            mode='bilinear',align_corners=False)[0]
        spatial_rows, legacy_rows, token_rows, valid_rows = [], [], [], []
        use_m = min(k, top_m)
        for y0, x0, y1, x1 in boxes:
            if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
                raise ValueError(f'Crop outside source image: {(y0,x0,y1,x1)}')
            local = prob[:, y0:y1, x0:x1]
            mean_prob = local.mean((1, 2))
            mean_top, ids = mean_prob.topk(use_m)
            local_winner = local.argmax(0)
            occupancies = torch.stack([(local_winner == cls).float().mean() for cls in ids])

            local_maps = torch.stack((
                local[ids[0]], local[ids[1]],
                top1[y0:y1, x0:x1], top2[y0:y1, x0:x1],
                entropy[y0:y1, x0:x1], margin[y0:y1, x0:x1],
                boundary[y0:y1, x0:x1], gradient[y0:y1, x0:x1],
            ), dim=0)
            if raw is not None:
                r1, r2 = raw[ids[0], y0:y1, x0:x1], raw[ids[1], y0:y1, x0:x1]
                rmax = raw[:,y0:y1,x0:x1].max(0).values
                raw_maps = torch.stack((r1,r2,r1-r2,rmax),dim=0)
                local_maps = torch.cat((local_maps,raw_maps),dim=0)
            pooled = F.adaptive_avg_pool2d(local_maps, (grid_size, grid_size))
            spatial_rows.append(pooled)

            disagreement = (local_winner != ids[0]).float().mean()
            h_ratio, w_ratio = (y1-y0)/height, (x1-x0)/width
            scalars = mean_top.new_tensor([
                h_ratio, w_ratio, (y0+y1)/(2*height), (x0+x1)/(2*width)
            ])
            scalars = torch.cat((
                mean_top[:2],
                (mean_top[0]-mean_top[1]).view(1),
                entropy[y0:y1, x0:x1].mean().view(1),
                disagreement.view(1), scalars,
            ))
            legacy_rows.append(torch.cat(((text[ids[0]]-text[ids[1]]).abs(), scalars)))

            tokens = torch.cat((text[ids], mean_top[:, None], occupancies[:, None]), dim=1)
            if use_m != top_m:
                tokens = F.pad(tokens, (0, 0, 0, top_m-use_m))
            valid = torch.arange(top_m, device=logits.device) < use_m
            token_rows.append(tokens)
            valid_rows.append(valid)

        out = {
            'spatial': torch.stack(spatial_rows),
            'legacy': torch.stack(legacy_rows),
            'class_tokens': torch.stack(token_rows),
            'class_valid': torch.stack(valid_rows),
        }
        assert out['spatial'].shape[1:] == (12 if raw is not None else 8, grid_size, grid_size)
        return {name: tensor.detach() for name, tensor in out.items()}
