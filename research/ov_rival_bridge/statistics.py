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
            rival_id, margins, active = m._rival(scores, valid)
            update = (output.float() - features.float())
            batch, cats, height, width = scores.shape
            s = scores.float().reshape(batch * cats, 1, height, width)
            score_margin = margins.reshape(batch * cats, 1, height, width) if m.arm != "S" else torch.zeros_like(s)
            valid_space = torch.ones_like(score_margin)
            mean = m._masked_local_mean(score_margin, valid_space)
            gate_input = torch.cat((s, score_margin, score_margin.abs(), (score_margin - mean).abs()), dim=1)
            gates = m.control(gate_input)
            accepted = torch.nn.functional.hardsigmoid(gates[:, :1])
            mixed = torch.sigmoid(gates[:, 1:2])
            switch_horizontal = (rival_id[:, :, :, 1:] != rival_id[:, :, :, :-1]).float().mean() if width > 1 else scores.new_zeros(())
            switch_vertical = (rival_id[:, :, 1:, :] != rival_id[:, :, :-1, :]).float().mean() if height > 1 else scores.new_zeros(())
            metrics = {
                "alpha": m.alpha.detach(),
                "update_rms": update.square().mean().sqrt(),
                "update_max": update.abs().max(),
                "margin_mean": margins.mean(),
                "margin_abs_mean": margins.abs().mean(),
                "rival_available": active.float().mean(),
                "gate_mean": accepted.mean(),
                "gate_zero_fraction": (accepted == 0).float().mean(),
                "gate_saturated_fraction": (accepted == 1).float().mean(),
                "mix_mean": mixed.mean(),
                "rival_switch_horizontal": switch_horizontal,
                "rival_switch_vertical": switch_vertical,
            }
            for name, value in metrics.items():
                storage.put_scalar("rival_bridge/" + name, value.item(), smoothing_hint=False)

    return module.register_forward_hook(record)
