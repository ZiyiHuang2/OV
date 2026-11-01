"""Bounded scalar monitoring of competition updates during training."""
import torch
from torch.nn import functional as F
from detectron2.utils.events import get_event_storage


def attach_statistics(module, period=100):
    def log_stats(m, inputs, output):
        if not m.training:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        if storage.iter % period:
            return
        with torch.no_grad():
            x, scores = inputs
            contrast = m.contrast(scores)
            gate = F.hardsigmoid(m.accept(torch.stack([scores.float(), contrast], dim=-1).to(m.accept[0].weight.dtype))).float()
            update = output.float() - x.float()
            values = {
                'contrast_min': contrast.min(), 'contrast_max': contrast.max(),
                'contrast_mean': contrast.mean(), 'contrast_std': contrast.std(),
                'gate_mean': gate.mean(), 'gate_zero_fraction': (gate == 0).float().mean(),
                'gate_saturated_fraction': ((gate == 0) | (gate == 1)).float().mean(),
                'alpha': m.alpha, 'update_rms': update.square().mean().sqrt(),
            }
            for name, value in values.items():
                storage.put_scalar('competition/' + name, value.item(), smoothing_hint=False)
    module.register_forward_hook(log_stats)
