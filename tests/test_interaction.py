"""Information paths and vocabulary contracts of the interaction prototype."""
import pytest
import torch

from cat_seg.engine.checkpoint import EvaluationCheckpointer, OVRSCheckpointer, model_config
from cat_seg.modeling.modules import CostConditionedGuidance, InteractionFusion, JointAffinityDecoder
from cat_seg.modeling.transformer.cat_seg_predictor import CATSegPredictor
from test_modules import inputs, make_model
from test_prototypes import config


FULL = dict(context_adapter='cost_lsk', aggregation='parallel_interaction',
            cad_stages=(1,), cad_scope='per_class', cad_mode='affinity')


@pytest.mark.parametrize('classes', [1, 3, 17])
def test_full_dynamic_classes_permutation_and_image_isolation(classes):
    torch.manual_seed(8)
    model = make_model(**FULL, pad_len=24).eval()
    image, text, guidance = inputs(batch=2, classes=classes)
    permutation = torch.randperm(classes)
    with torch.no_grad():
        output = model(image, text, guidance)
        reordered = model(image, text[:, permutation], guidance)
        individual = model(image[:1], text[:1], [g[:1] for g in guidance])
    assert output.shape == (2, classes, 32, 32)
    torch.testing.assert_close(reordered, output[:, permutation], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(individual, output[:1], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize('options', [dict(context_adapter='cost_lsk'),
                                    dict(aggregation='parallel_interaction'),
                                    dict(cad_stages=(1,), cad_scope='per_class', cad_mode='affinity'), FULL])
def test_new_parameters_receive_finite_nonzero_gradients(options):
    model = make_model(**options).train()
    model(*inputs(batch=2, classes=3)).square().mean().backward()
    for name, parameter in model.named_parameters():
        if any(part in name for part in ('cost_context', '.fusion.', 'guidance_refiner')):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_m1_reads_current_cost_and_m2_output_reaches_next_m1():
    module = CostConditionedGuidance(16, 8).eval()
    guide, cost = torch.randn(2, 8, 6, 6), torch.randn(2, 16, 3, 6, 6)
    assert not torch.allclose(module(guide, cost), module(guide, torch.randn_like(cost)), rtol=0, atol=1e-7)
    model = make_model(**FULL).eval()
    events = {}
    handles = [
        model.layers[0].fusion.register_forward_hook(lambda m, a, o: events.update(fused=o.detach().clone())),
        model.layers[1].cost_context.register_forward_pre_hook(lambda m, a: events.update(next_cost=a[1].detach().clone())),
    ]
    try:
        model(*inputs())
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(events['next_cost'], events['fused'], rtol=0, atol=0)


def test_m2_routing_is_conditioned_on_visual_and_text_guidance():
    module = InteractionFusion(16, 8, 8).eval()
    s, c = torch.randn(1, 16, 3, 6, 6), torch.randn(1, 16, 3, 6, 6)
    guide, text = torch.randn(1, 8, 6, 6), torch.randn(1, 3, 8)
    output = module(s, c, guide, text)
    assert not torch.allclose(output, module(s, c, torch.randn_like(guide), text), rtol=0, atol=1e-7)
    assert not torch.allclose(output, module(s, c, guide, torch.randn_like(text)), rtol=0, atol=1e-7)


def test_affinity_boundary_normalization_preserves_constant_values():
    module = JointAffinityDecoder(16, 8, 4).eval()
    keys, query = torch.randn(2, 8, 3, 4), torch.randn(2, 8, 6, 8)
    value = torch.full((2, 8, 3, 4), 2.5)
    result = module.reassemble(keys, query, value)
    torch.testing.assert_close(result, torch.full_like(result, 2.5), rtol=1e-6, atol=1e-6)


def test_m3_class_order_and_batch_order_match_individual_calls():
    module = JointAffinityDecoder(16, 8, 4).eval()
    low, up, guide = torch.randn(6, 16, 3, 4), torch.randn(6, 8, 6, 8), torch.randn(2, 4, 6, 8)
    cost, refined = module(low, up, guide)
    outputs = [module(low[i:i+1], up[i:i+1], guide[i // 3:i // 3 + 1]) for i in range(6)]
    torch.testing.assert_close(cost, torch.cat([o[0] for o in outputs]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(refined, torch.cat([o[1] for o in outputs]), rtol=1e-5, atol=1e-6)
    # Refined guidance is actually used for cost reassembly, not just concatenation.
    with torch.no_grad():
        for parameter in module.guidance_refiner.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    changed_cost, _ = module(low, up, guide)
    assert not torch.allclose(cost, changed_cost, rtol=0, atol=1e-7)


def test_zero_residual_scales_recover_known_anchors():
    m1 = CostConditionedGuidance(16, 8, residual_scale=0)
    guide, cost = torch.randn(1, 8, 6, 6), torch.randn(1, 16, 3, 6, 6)
    torch.testing.assert_close(m1(guide, cost), guide, rtol=0, atol=0)
    m2 = InteractionFusion(16, 8, 8, residual_scale=0)
    category = torch.randn_like(cost)
    torch.testing.assert_close(m2(cost, category, guide, torch.randn(1, 3, 8)),
                               0.5 * (cost + category), rtol=0, atol=0)
    m3 = JointAffinityDecoder(16, 8, 4, residual_scale=0)
    low, up, guide = torch.randn(3, 16, 3, 4), torch.randn(3, 8, 6, 8), torch.randn(1, 4, 6, 8)
    actual, refined = m3(low, up, guide)
    torch.testing.assert_close(actual, up, rtol=0, atol=0)
    torch.testing.assert_close(refined, guide.expand(3, -1, -1, -1), rtol=0, atol=0)


@pytest.mark.parametrize('experiment', ['interaction_guidance', 'interaction_fusion',
                                      'interaction_decoder', 'full_interaction'])
def test_experiment_reaches_predictor_and_configuration_contract(experiment):
    cfg = config(experiment)
    options = CATSegPredictor.from_config(cfg)
    signature = model_config(cfg)
    assert options['context_adapter'] == cfg.MODEL.OVRS.CONTEXT_ADAPTER
    assert options['aggregation'] == cfg.MODEL.OVRS.AGGREGATION
    assert options['cad_mode'] == cfg.MODEL.OVRS.CAD_MODE
    if options['cad_mode'] == 'affinity':
        assert signature['MODEL.OVRS.CAD_MODE'] == 'affinity'
    if options['context_adapter'] == 'cost_lsk':
        assert signature['MODEL.OVRS.LSK_DILATION'] == 3


@pytest.mark.parametrize('options', [dict(cad_mode='affinity'),
                                    dict(cad_stages=(1,), cad_mode='affinity')])
def test_affinity_config_rejects_incompatible_scope(options):
    with pytest.raises(ValueError, match='affinity CAD requires'):
        make_model(**options)


def test_cpu_bfloat16_full_forward_backward():
    model = make_model(**FULL)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(*inputs(classes=3))
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_affinity_checkpoint_rejects_residual_computation_before_loading(tmp_path):
    source, target = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    cfg = config('full_interaction')
    OVRSCheckpointer(source, str(tmp_path), expected_config=model_config(cfg)).save('model')
    before = target.weight.detach().clone()
    cfg.MODEL.OVRS.CAD_MODE = 'residual'
    with pytest.raises(ValueError, match='configuration mismatch'):
        EvaluationCheckpointer(target, expected_config=model_config(cfg)).load(str(tmp_path / 'model.pth'))
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_interaction_operates_on_selected_real_classes_before_decoder():
    model = make_model(**FULL, pad_len=5).eval()
    output = model(*inputs(classes=9))
    assert output.shape == (1, 9, 32, 32)
    assert ((output == -100).flatten(2).all(dim=-1).sum(dim=1) == 4).all()
