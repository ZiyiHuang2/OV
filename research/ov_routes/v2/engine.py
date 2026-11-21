"""Exact-budget Route-B V2 inference and source-only crop-benefit supervision.

The full CAT-Seg host stays frozen. The selector is never given a GT mask
at prediction time. The crop coordinates and fusion match Route B V1.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F

from .evidence import build_evidence, windows
from .diagnostics import gt_present_miou


def fuse_probabilities(base: torch.Tensor, crop_results):
    """Exact V1 fusion: mean of sigmoid scores over overlapping predictions."""
    if base.ndim != 3:
        raise ValueError('Expected [K,H,W] logits')
    result = base.sigmoid().clone()
    count = result.new_ones(result.shape[-2:])
    for (y0,x0,y1,x1), crop in crop_results:
        if crop.shape != (len(base),y1-y0,x1-x0):
            raise ValueError('Unaligned crop score dimensions')
        result[:,y0:y1,x0:x1] += crop.sigmoid()
        count[y0:y1,x0:x1] += 1
    return result/count


@torch.no_grad()
def first_pass(host, image: torch.Tensor, names: list[str], *, with_raw_cost: bool):
    """Optionally capture the genuine pre-aggregation CLIP correlation maps.

    CAT-Seg's Aggregator.corr_embed calls transformer.conv1 on a [K,P,h,w]
    tensor of CLIP image/text dot products. The pre-hook observes that tensor,
    without changing the original module, outputs or pretrained weights.
    """
    if not with_raw_cost:
        return host.predict_logits(image[None],names)[0], None
    aggregator = host.predictor.transformer
    layer = aggregator.conv1
    captured = []
    handle = layer.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    try:
        logits = host.predict_logits(image[None],names)[0]
    finally:
        handle.remove()
    if len(captured)!=1 or captured[0].ndim!=4 or captured[0].shape[0]!=len(names):
        raise RuntimeError('CAT-Seg conv1 must receive exactly one [K,P,h,w] CLIP cost tensor')
    raw_cost = captured[0].mean(dim=1)
    return logits, raw_cost


def batch_evidence(evidence: Dict[str,torch.Tensor]):
    return {key: value.unsqueeze(0) for key,value in evidence.items()}


@torch.no_grad()
def training_example(host, image: torch.Tensor, labels: torch.Tensor, names: list[str],
                     *, grid_size: int = 8, top_m: int = 4,
                     with_raw_cost: bool = False):
    """GT only creates supervision on SOURCE TRAIN images.

    Compute one base encoding and eight TRUE crop encodings (nine total),
    not cache-feature ROI pooling. Gains reproduce original V1 training_actions
    definition: whole-image valid-pixel BCE decrease after one-crop fusion.
    """
    if image.ndim != 3 or labels.shape != image.shape[-2:]:
        raise ValueError('Image CHW and label HW shapes must match')
    valid = labels != 255
    if not valid.any():
        return None
    if ((labels[valid]<0)|(labels[valid]>=len(names))).any():
        raise ValueError('Invalid SOURCE class target: unseen/ignored GT leak risk')
    boxes = windows(*image.shape[-2:])
    base, raw_cost = first_pass(host,image,names,with_raw_cost=with_raw_cost)
    evidence = build_evidence(base,host.text_vectors(names),boxes,
                              grid_size=grid_size,top_m=top_m,raw_cost=raw_cost)
    base_prob = base.sigmoid()
    gains, iou_gains = [], []
    # A fixed GT-present class set gives comparable per-image IoU proxy across
    # the eight actions; this is diagnostics only, never a selector input.
    base_pred = base_prob.argmax(0)
    iou0 = gt_present_miou(base_pred, labels, len(names))
    for y0,x0,y1,x1 in boxes:
        inside_valid = valid[y0:y1,x0:x1]
        if not inside_valid.any():
            # Preserve original V1 semantic; no label-supported action benefit.
            gains.append(base.new_zeros(()))
            iou_gains.append(base.new_zeros(()))
            continue
        crop = host.predict_logits(image[None,:,y0:y1,x0:x1].contiguous(),names)[0]
        old = base_prob[:,y0:y1,x0:x1].permute(1,2,0)[inside_valid]
        new = ((base_prob[:,y0:y1,x0:x1]+crop.sigmoid())/2).permute(1,2,0)[inside_valid]
        gt = F.one_hot(labels[y0:y1,x0:x1][inside_valid].long(),len(names)).float()
        gain = (F.binary_cross_entropy(old,gt,reduction='sum')-
                F.binary_cross_entropy(new,gt,reduction='sum')) / (valid.sum()*len(names))
        gains.append(gain)
        # The crop alters only its own spatial support under V1 fusion.
        pred = base_pred.clone()
        pred[y0:y1,x0:x1] = ((base_prob[:,y0:y1,x0:x1] + crop.sigmoid())/2).argmax(0)
        iou_gains.append((gt_present_miou(pred, labels, len(names)) - iou0).to(base))
    return {'evidence': {k:v.cpu().contiguous() for k,v in evidence.items()},
            'gains': torch.stack(gains).cpu().contiguous(),
            'iou_gains': torch.stack(iou_gains).cpu().contiguous(),
            'meta': {'num_candidates':len(boxes),'source_only':True}}


@torch.no_grad()
def predict(host, image: torch.Tensor, names: list[str], selector,
            *, grid_size: int = 8, top_m: int = 4, allow_skip: bool = False,
            scores_are_gain: bool = True):
    """Exactly 1 + {0,1} full host calls; no GT argument, no Oracle use.

    ``allow_skip`` applies the prespecified zero BCE-benefit threshold to
    regression-calibrated selector outputs. Disabled for first V2 ablation.
    """
    if image.ndim != 3:
        raise ValueError('Expected CHW RGB')
    base, raw_cost = first_pass(host,image,names,with_raw_cost=selector.use_raw_cost if hasattr(selector,"use_raw_cost") else False)
    boxes = windows(*image.shape[-2:])
    evidence = build_evidence(base,host.text_vectors(names),boxes,
                              grid_size=grid_size,top_m=top_m,raw_cost=raw_cost)
    selector.eval()
    scores = selector(batch_evidence(evidence))[0]
    chosen = int(scores.argmax().item())
    if allow_skip and not scores_are_gain:
        raise ValueError('Skip requires learned calibrated gain predictions')
    skip = bool(allow_skip and scores[chosen].item() <= 0.0)
    score_meta={'selection_scores':scores.cpu().tolist(),
                'score_semantics':'scaled_bce_gain' if scores_are_gain else 'heuristic_rank'}
    if skip:
        return base.sigmoid(), {
            'selected': [],'encoder_calls':1,'skip':True,
            **score_meta,'boxes':[],
        }
    box = boxes[chosen]
    y0,x0,y1,x1 = box
    crop = host.predict_logits(image[None,:,y0:y1,x0:x1].contiguous(),names)[0]
    fused = fuse_probabilities(base,[(box,crop)])
    return fused, {'selected':[chosen], 'encoder_calls':2,'skip':False,
                   **score_meta, 'boxes':[list(box)]}
