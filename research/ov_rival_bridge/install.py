"""Attach the experimental module after loading common full CAT-Seg weights.

Must run before optimizer/DDP construction, with model parameters on device.
"""
from .bridge import RivalConditionedTransfer


def install(model, arm, *, initial_scale=0.01, rank=24, attn_dim=8):
    if arm not in ("S", "M", "R", "P"):
        raise ValueError("arm must be S/M/R/P")
    aggregator = model.sem_seg_head.predictor.transformer
    if not hasattr(aggregator, "layers") or not aggregator.layers:
        raise ValueError("Expected a CAT-Seg aggregator with layers")
    first = aggregator.layers[0]
    if getattr(first, "aggregation", "serial") != "serial":
        raise ValueError("Rival bridge supports native serial aggregation only")
    if getattr(first, "cost_context", None) is not None:
        raise ValueError("Cannot combine with cost-context replacement")
    if getattr(aggregator, "competition_update", None) is not None:
        raise ValueError("Cannot install rival and Competition V1 together")
    if getattr(aggregator, "rival_update", None) is not None:
        raise ValueError("Rival bridge already installed")
    dim = aggregator.hidden_dim
    aggregator.rival_update = RivalConditionedTransfer(
        dim, arm=arm, rank=rank, attn_dim=attn_dim, init_scale=initial_scale
    ).to(next(model.parameters()).device)
    return model
