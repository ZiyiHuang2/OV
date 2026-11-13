"""Deterministic isolation tests for the RCT Value-only hypothesis."""
import copy
import pytest
import torch

from .bridge import RivalConditionedTransfer


def example(b=2, d=12, k=4, h=5, w=6):
    torch.manual_seed(13)
    x = torch.randn(b, d, k, h, w, requires_grad=True)
    scores = torch.randn(b, k, h, w, requires_grad=True)
    scores = scores + torch.arange(k).reshape(1, k, 1, 1) * 0.15
    return x, scores


@pytest.mark.parametrize("arm", ["S", "M", "V", "R", "P"])
def test_forward_backward_active_paths(arm):
    x, s = example()
    module = RivalConditionedTransfer(12, arm, rank=5, attn_dim=3)
    y, aux = module(x, s, return_aux=True)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert module.alpha.grad is not None and module.alpha.grad.abs().sum() > 0
    assert module.out.weight.grad is not None and module.out.weight.grad.abs().sum() > 0
    assert module.control[2].weight.grad is not None
    assert module.control[2].weight.grad.abs().sum() > 0
    if arm != "P":
        for param in (module.query.weight, module.key.weight, module.pair[2].weight):
            assert param.grad is not None and param.grad.abs().sum() > 0
    if arm in {"V", "R"}:
        assert module.control[2].weight.grad[1].abs().sum() > 0
    else:
        assert module.control[2].weight.grad[1].abs().sum() == 0
    assert 0 <= aux["gate_mean"] <= 1


@pytest.mark.parametrize("arm", ["S", "M", "V", "R", "P"])
def test_class_equivariance_and_single_class(arm):
    x, s = example(b=1)
    model = RivalConditionedTransfer(12, arm, rank=6)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        baseline = model(x, s)
        permuted = model(x[:, :, perm], s[:, perm])
    torch.testing.assert_close(permuted, baseline[:, :, perm], rtol=1e-5, atol=1e-5)
    x1, s1 = example(b=1, k=1)
    y1 = model(x1, s1)
    if arm in {"V", "R", "P"}:
        assert torch.equal(y1, x1)


def test_value_only_has_exact_margin_attention():
    x, s = example(b=1)
    model_m = RivalConditionedTransfer(12, "M", rank=6)
    model_v = copy.deepcopy(model_m)
    model_v.arm = "V"
    records_m, records_v = [], []
    def hook(records):
        return lambda _, args, output: records.append((args[0].detach().clone(), output.detach().clone()))
    handles = []
    for mod_m, mod_v in ((model_m.query, model_v.query),
                         (model_m.key, model_v.key), (model_m.pair, model_v.pair)):
        handles.append(mod_m.register_forward_hook(hook(records_m)))
        handles.append(mod_v.register_forward_hook(hook(records_v)))
    with torch.no_grad():
        out_m = model_m(x, s)
        out_v = model_v(x, s)
    for h in handles:
        h.remove()
    for (input_m, output_m), (input_v, output_v) in zip(records_m, records_v):
        torch.testing.assert_close(input_m, input_v, rtol=0, atol=0)
        torch.testing.assert_close(output_m, output_v, rtol=0, atol=0)
    assert (out_m - out_v).abs().max().item() > 1e-7


def test_invalid_spatial_changes_do_not_affect_valid_outputs():
    x, s = example(b=1)
    model = RivalConditionedTransfer(12, "R", rank=5)
    valid = torch.ones(1, 5, 6, dtype=torch.bool)
    valid[:, :, -1] = False
    with torch.no_grad():
        baseline = model(x, s, spatial_mask=valid)
        x2 = x.detach().clone()
        s2 = s.detach().clone()
        x2[:, :, :, :, -1] += torch.randn_like(x2[:, :, :, :, -1]) * 2500
        s2[:, :, :, -1] += torch.randn_like(s2[:, :, :, -1]) * 2500
        changed = model(x2, s2, spatial_mask=valid)
    expanded = valid[:, None, None].expand_as(baseline)
    torch.testing.assert_close(baseline[expanded], changed[expanded], rtol=1e-5, atol=1e-5)
    assert torch.equal(baseline[:, :, :, :, -1], x[:, :, :, :, -1])


def test_invalid_class_does_not_affect_valid_outputs():
    x, s = example(b=1)
    mask = torch.tensor([True, True, False, True])
    model = RivalConditionedTransfer(12, "R", rank=5)
    with torch.no_grad():
        baseline = model(x, s, class_mask=mask)
        x2, s2 = x.detach().clone(), s.detach().clone()
        x2[:, :, 2] += 2500
        s2[:, 2] += 2500
        changed = model(x2, s2, class_mask=mask)
    torch.testing.assert_close(baseline[:, :, mask], changed[:, :, mask], rtol=1e-5, atol=1e-5)
    assert torch.equal(baseline[:, :, 2], x[:, :, 2])


def test_exact_bypass_and_optional_debug_checks():
    x, s = example()
    model = RivalConditionedTransfer(12, "R")
    assert torch.equal(model(x, s, enabled=False), x)
    model.debug_finite = True
    x2 = x.detach().clone()
    x2[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(x2, s)
