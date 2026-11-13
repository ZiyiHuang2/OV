"""Bounded optional telemetry for rival-conditioned feature transfer."""
import torch
from detectron2.utils.events import get_event_storage


def attach_statistics(module, period=100):
    if period < 1:
        raise ValueError("period must be positive")

    def record(m, inputs, output):
        if not m.training:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        if storage.iter % period:
            return
        with torch.no_grad():
            features, scores = inputs[:2]
            valid = m._class_valid(scores, None)
            _, margins, active = m._rival(scores, valid)
            update = (output.float() - features.float())
            metrics = {
                "alpha": m.alpha.detach(),
                "update_rms": update.square().mean().sqrt(),
                "update_max": update.abs().max(),
                "margin_mean": margins.mean(),
                "margin_abs_mean": margins.abs().mean(),
                "rival_available": active.float().mean(),
            }
            for name, value in metrics.items():
                storage.put_scalar("rival_bridge/" + name, value.item(), smoothing_hint=False)

    return module.register_forward_hook(record)
