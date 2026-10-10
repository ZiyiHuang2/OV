import pytest
import torch

from cat_seg.modeling.transformer.model import Aggregator, AggregatorLayer
from cat_seg.modeling.transformer.ovrs_modules import CostGuidanceRefiner, LSKGuidanceAdapter, ParallelFusion
from cat_seg.modeling.losses import masked_segmentation_bce


def make_model(**options):
    return Aggregator(
        text_guidance_dim=32, text_guidance_proj_dim=16,
        appearance_guidance_dim=32, appearance_guidance_proj_dim=16,
        decoder_dims=(32, 16), decoder_guidance_dims=(16, 8),
        decoder_guidance_proj_dims=(8, 4), hidden_dim=32,
        num_layers=2, nheads=4, pooling_size=(2, 2),
        feature_resolution=(8, 8), window_size=4, pad_len=options.pop('pad_len', 0), **options,
    )


def inputs(batch=1, classes=3):
    return (torch.randn(batch, 32, 8, 8), torch.randn(batch, classes, 1, 32),
            [torch.randn(batch, 32, 8, 8), torch.randn(batch, 16, 16, 16),
             torch.randn(batch, 8, 32, 32)])


@pytest.mark.parametrize('options', [
    {}, {'aggregation': 'parallel_simple'}, {'aggregation': 'parallel_epl'},
    {'context_adapter': 'lsk'}, {'cad_stages': (1,)},
    {'aggregation': 'parallel_epl', 'context_adapter': 'lsk', 'cad_stages': (1,)},
    {'aggregation': 'parallel_epl', 'context_adapter': 'lsk', 'cad_stages': (1, 2)},
])
def test_forward_backward(options):
    model = make_model(**options).train()
    output = model(*inputs(batch=2))
    assert output.shape == (2, 3, 32, 32)
    output.square().mean().backward()
    for name, parameter in model.named_parameters():
        if any(part in name for part in ('context_adapter', 'fusion', 'guidance_refiner')):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize('classes', [1, 3, 17])
def test_dynamic_vocabulary_and_permutation(classes):
    model = make_model(aggregation='parallel_epl', context_adapter='lsk', cad_stages=(1,), pad_len=24).eval()
    image, text, guidance = inputs(classes=classes)
    permutation = torch.randperm(classes)
    with torch.no_grad():
        original = model(image, text, guidance)
        reordered = model(image, text[:, permutation], guidance)
    assert original.shape[1] == classes
    torch.testing.assert_close(reordered, original[:, permutation], atol=2e-5, rtol=2e-5)


def test_cad_does_not_mix_images_and_is_category_order_invariant():
    refiner = CostGuidanceRefiner(12, 8)
    cost = torch.randn(2, 3, 12, 16, 16)
    guide = torch.randn(2, 8, 16, 16)
    combined = refiner(cost.reshape(6, 12, 16, 16), guide)
    separate = torch.cat([refiner(cost[i], guide[i:i + 1]) for i in range(2)])
    reordered = refiner(cost[:, [2, 0, 1]].reshape(6, 12, 16, 16), guide)
    torch.testing.assert_close(combined, separate)
    torch.testing.assert_close(combined, reordered)


def test_lsk_zero_residual_is_identity():
    adapter = LSKGuidanceAdapter(16, residual_scale=0)
    guide = torch.randn(2, 16, 24, 24)
    torch.testing.assert_close(adapter(guide), guide, rtol=0, atol=0)


def test_ignored_pixels_have_zero_gradient_and_no_loss_contribution():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    labels[:, :2] = 255
    first = masked_segmentation_bce(logits, labels)
    altered = logits.detach().clone()
    altered[:, :, :2] += 100
    torch.testing.assert_close(first, masked_segmentation_bce(altered, labels))
    first.backward()
    assert logits.grad[:, :, :2].abs().sum() == 0
    assert logits.grad[:, :, 2:].abs().sum() > 0


def test_all_ignore_is_finite_differentiable_zero():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    loss = masked_segmentation_bce(logits, torch.full((1, 4, 4), 255))
    loss.backward()
    assert loss.item() == 0
    assert logits.grad.abs().sum() == 0


@pytest.mark.parametrize('mode', ['simple', 'epl'])
def test_zero_scale_fusion_retains_branch_mean(mode):
    fusion = ParallelFusion(16, mode=mode, residual_scale=0)
    spatial, category = torch.randn(2, 16, 3, 8, 8), torch.randn(2, 16, 3, 8, 8)
    torch.testing.assert_close(fusion(spatial, category), (spatial + category) / 2, rtol=0, atol=0)


def test_zero_scale_cad_retains_guidance():
    refiner = CostGuidanceRefiner(12, 8, residual_scale=0)
    guide = torch.randn(2, 8, 16, 16)
    torch.testing.assert_close(refiner(torch.randn(6, 12, 16, 16), guide), guide, rtol=0, atol=0)


def test_parallel_branches_read_same_cost_once():
    class RecordingBranch(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset, self.calls = offset, []

        def forward(self, cost, guidance):
            self.calls.append(cost)
            return cost + self.offset

    layer = AggregatorLayer(hidden_dim=32, appearance_guidance=16,
                            text_guidance_dim=16, input_resolution=(8, 8),
                            window_size=4, aggregation='parallel_epl', fusion_scale=0)
    layer.swin_block, layer.attention = RecordingBranch(2), RecordingBranch(4)
    cost = torch.randn(2, 32, 3, 8, 8)
    output = layer(cost, None, None)
    assert len(layer.swin_block.calls) == len(layer.attention.calls) == 1
    assert layer.swin_block.calls[0] is cost and layer.attention.calls[0] is cost
    torch.testing.assert_close(output, cost + 3)


def test_full_model_wires_shared_guidance_and_real_class_count():
    model = make_model(aggregation='parallel_epl', context_adapter='lsk', cad_stages=(1,), pad_len=5)
    events = []
    def record(name):
        return lambda module, args, output: events.append((name, [tuple(x.shape) for x in args]))
    handles = [model.context_adapter.register_forward_hook(record('lsk'))]
    handles += [layer.fusion.register_forward_hook(record('fusion')) for layer in model.layers]
    handles.append(model.decoder1.guidance_refiner.register_forward_hook(record('cad')))
    model(*inputs(batch=2, classes=3))
    for handle in handles:
        handle.remove()
    assert [name for name, shapes in events] == ['lsk', 'fusion', 'fusion', 'cad']
    assert events[0][1] == [(2, 16, 8, 8)]
    assert events[1][1] == [(2, 32, 3, 8, 8)] * 2
    assert events[3][1] == [(6, 24, 16, 16), (2, 8, 16, 16)]
