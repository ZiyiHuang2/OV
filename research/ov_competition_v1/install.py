"""Attach B/C to an *already instantiated / pretrained-weight-loaded* CAT-Seg.

Requires the repository-specific Aggregator.forward adapter from patch_repo.py.
Do this BEFORE constructing optimizer/DDP. A should not attach anything.
"""
from .competitive_update import CompetitiveSpatialUpdate


def install(model, arm, *, initial_scale=0.01, hidden=24):
    if arm not in ("A", "B", "C"):
        raise ValueError("arm must be A/B/C")
    aggregator = model.sem_seg_head.predictor.transformer
    if not hasattr(aggregator, "layers") or not aggregator.layers:
        raise ValueError("Expected a CAT-Seg aggregator with layers")
    first = aggregator.layers[0]
    if getattr(first, "aggregation", "serial") != "serial":
        raise ValueError("V1 adapter supports only native serial aggregation")
    if getattr(first, "cost_context", None) is not None:
        raise ValueError("V1 adapter does not support a cost-context replacement")
    if arm == "A":
        aggregator.competition_update = None
        return model
    dim = aggregator.hidden_dim
    aggregator.competition_update = CompetitiveSpatialUpdate(
        dim, "self_only" if arm == "B" else "competitive",
        hidden=hidden, initial_scale=initial_scale)
    aggregator.competition_update.to(next(model.parameters()).device)
    return model
