import pytest
import torch

from cat_seg.modeling.modules import CounterfactualEvidenceVerifier


def test_crtv_is_identity_at_zero_scale():
    verifier = CounterfactualEvidenceVerifier(residual_scale=0.0)
    correlation = torch.randn(2, 3, 5, 4, 4)
    text = torch.randn(2, 5, 3, 8)
    output, diagnostics = verifier(correlation, text)
    torch.testing.assert_close(output, correlation, rtol=0, atol=0)
    assert diagnostics["margin"].shape == (2, 5, 4, 4)
    assert diagnostics["relation"].shape == (2, 5, 5)


@pytest.mark.parametrize("classes", [1, 2, 5, 17])
def test_crtv_supports_dynamic_vocabulary_and_permutation(classes):
    # A singleton vocabulary has no competitors; it must still be finite.
    verifier = CounterfactualEvidenceVerifier(residual_scale=0.2)
    correlation = torch.randn(2, 2, classes, 3, 3)
    text = torch.randn(2, classes, 2, 8)
    order = torch.randperm(classes)
    original, _ = verifier(correlation, text)
    reordered, _ = verifier(correlation[:, :, order], text[:, order])
    torch.testing.assert_close(reordered, original[:, :, order], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(reordered).all()


def test_crtv_learns_only_through_bounded_relation_residual():
    verifier = CounterfactualEvidenceVerifier(residual_scale=0.0)
    correlation = torch.randn(2, 2, 4, 3, 3)
    text = torch.randn(2, 4, 2, 8)
    output, _ = verifier(correlation, text)
    output.square().mean().backward()
    assert verifier.residual_scale.grad is not None
    assert torch.isfinite(verifier.residual_scale.grad).all()


def test_crtv_excludes_self_relation_and_topk_is_finite():
    verifier = CounterfactualEvidenceVerifier(residual_scale=0.5, topk=1)
    correlation = torch.randn(1, 1, 4, 2, 2)
    text = torch.randn(1, 4, 1, 16)
    _, diagnostics = verifier(correlation, text)
    relation = diagnostics["relation"]
    diagonal = relation[0].diagonal()
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)
    assert torch.isfinite(relation).all()
