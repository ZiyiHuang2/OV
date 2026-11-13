"""CPU tests exercise the actual module, not a mock host."""
import pytest
import torch
from .bridge import RivalConditionedTransfer


def sample(batch=2, dim=12, categories=4, height=5, width=6):
    torch.manual_seed(14)
    x = torch.randn(batch, dim, categories, height, width, requires_grad=True)
    s = torch.randn(batch, categories, height, width, requires_grad=True)
    s = s + torch.arange(categories)[None, :, None, None] * 0.125
    return x, s


@pytest.mark.parametrize('arm', ['S', 'M', 'R', 'P'])
def test_shape_finite_and_grad(arm):
    x, s = sample()
    module = RivalConditionedTransfer(12, arm=arm, rank=5, attn_dim=3)
    y, stats = module(x, s, return_aux=True)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert not stats['bypassed']
    y.square().mean().backward()
    assert module.alpha.grad is not None and torch.isfinite(module.alpha.grad)
    assert module.out.weight.grad is not None
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('arm', ['S', 'M', 'R', 'P'])
def test_class_permutation_and_dynamic_classes(arm):
    model = RivalConditionedTransfer(12, arm=arm, rank=5, attn_dim=3).eval()
    x, s = sample(batch=1)
    perm = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        original = model(x, s)
        changed = model(x[:, :, perm], s[:, perm])
    torch.testing.assert_close(changed, original[:, :, perm], rtol=1e-5, atol=2e-5)
    x1, s1 = sample(batch=1, categories=1)
    assert model(x1, s1).shape == x1.shape


def test_explicit_bypass_exact_and_single_class_rival_identity():
    model = RivalConditionedTransfer(12, arm='R')
    x, s = sample()
    assert torch.equal(model(x, s, enabled=False), x)
    x1, s1 = sample(categories=1)
    assert torch.equal(model(x1, s1), x1)


def test_invalid_class_spatial_masks_unchanged_and_finite():
    x, s = sample()
    model = RivalConditionedTransfer(12, arm='R', rank=6)
    cmask = torch.tensor([True, True, False, True])
    smask = torch.ones(2, 5, 6, dtype=torch.bool)
    smask[:, -1] = False
    y = model(x, s, cmask, smask)
    torch.testing.assert_close(y[:, :, 2], x[:, :, 2], rtol=0, atol=0)
    torch.testing.assert_close(y[:, :, :, -1], x[:, :, :, -1], rtol=0, atol=0)
    assert torch.isfinite(y).all()


def test_padding_class_does_not_change_real_class_outputs():
    x, s = sample(batch=1, categories=3)
    model = RivalConditionedTransfer(12, arm='R', rank=5)
    more_x = torch.cat([x, torch.randn_like(x[:, :, :1])], dim=2)
    more_s = torch.cat([s, torch.full_like(s[:, :1], 1000.)], dim=1)
    with torch.no_grad():
        reference = model(x, s)
        padded = model(more_x, more_s, torch.tensor([True,True,True,False]))
    torch.testing.assert_close(reference, padded[:, :, :3], rtol=1e-5, atol=2e-5)


def test_rival_feature_information_is_not_merely_a_score_margin():
    x, s = sample(batch=1, categories=3)
    model = RivalConditionedTransfer(12, arm='R', rank=5)
    with torch.no_grad():
        y1 = model(x, s)
        changed = x.detach().clone()
        changed[:, :, 1] = changed[:, :, 1] + torch.randn_like(changed[:, :, 1]) * 0.2
        y2 = model(changed, s)
    assert (y2[:, :, 0] - y1[:, :, 0]).abs().max().item() > 1e-8


def test_invalid_inputs_fail_closed():
    x, s = sample()
    model = RivalConditionedTransfer(12)
    with pytest.raises(ValueError):
        model(x, s, class_mask=torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError):
        model(x, s, spatial_mask=torch.ones(2, 5, 6))
    with pytest.raises(ValueError):
        model(x, s[:, :3])


def test_relation_gate_nontrivial_on_random_input():
    x, s = sample(batch=1)
    model = RivalConditionedTransfer(12, arm='R', rank=5)
    y, stat = model(x, s, return_aux=True)
    assert (y-x).abs().sum() > 0
    assert 0. < stat['gate_mean'] < 1.
    assert 0. < stat['mix_mean'] < 1.
