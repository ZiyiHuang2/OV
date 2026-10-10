"""Contracts for EPL routing and class-conditioned CAD in the real model path."""
import pytest
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from cat_seg import add_cat_seg_config
from cat_seg.engine import Trainer
from cat_seg.engine.checkpoint import EvaluationCheckpointer, OVRSCheckpointer, model_config
from cat_seg.modeling.modules import ClassGuidanceRefiner, CostGuidanceRefiner, ParallelFusion
from cat_seg.modeling.transformer.model import Up
from cat_seg.modeling.transformer.cat_seg_predictor import CATSegPredictor
from test_modules import inputs, make_model


FULL = dict(aggregation='parallel_epl', context_adapter='lsk', cad_stages=(1,))
VARIANTS = [dict(epl_routing='uniform'), dict(cad_scope='per_class'),
            dict(epl_routing='uniform', cad_scope='per_class')]


def config(experiment='full'):
    from pathlib import Path
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs' / 'ovrs' / (experiment + '.yaml')))
    return cfg


@pytest.mark.parametrize('options', VARIANTS)
def test_prototype_initial_parameters_match_full_and_gradients_are_finite(options):
    torch.manual_seed(42)
    reference = make_model(**FULL)
    torch.manual_seed(42)
    model = make_model(**FULL, **options)
    assert list(reference.state_dict()) == list(model.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    output = model(*inputs(batch=2, classes=3))
    assert output.shape == (2, 3, 32, 32)
    output.square().mean().backward()
    for name, parameter in model.named_parameters():
        if any(part in name for part in ('context_adapter', 'fusion', 'guidance_refiner')):
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().sum() > 0, name
            else:
                assert '.fusion.gate.' in name and parameter.grad is None


@pytest.mark.parametrize('routing', ['learned', 'uniform'])
@pytest.mark.parametrize('classes', [1, 3, 17])
def test_per_class_cad_full_model_dynamic_classes_and_permutation(routing, classes):
    model = make_model(**FULL, epl_routing=routing, cad_scope='per_class', pad_len=24).eval()
    image, text, guide = inputs(batch=2, classes=classes)
    order = torch.randperm(classes)
    with torch.no_grad():
        original = model(image, text, guide)
        reordered = model(image, text[:, order], guide)
    assert original.shape == (2, classes, 32, 32)
    torch.testing.assert_close(reordered, original[:, order], atol=2e-5, rtol=2e-5)


def test_uniform_epl_is_dense_and_does_not_execute_or_optimize_router():
    torch.manual_seed(7)
    learned = ParallelFusion(16, mode='epl')
    torch.manual_seed(7)
    uniform = ParallelFusion(16, mode='epl', epl_routing='uniform')
    def forbidden(*args):
        raise AssertionError('uniform routing must not execute the router')
    handle = uniform.gate.register_forward_pre_hook(forbidden)
    # Force the learned router to produce the same equal weights as the control.
    with torch.no_grad():
        learned.gate[-2].weight.zero_()
        learned.gate[-2].bias.zero_()
    spatial, category = torch.randn(2, 16, 3, 8, 8), torch.randn(2, 16, 3, 8, 8)
    output = uniform(spatial, category)
    torch.testing.assert_close(output, learned(spatial, category), rtol=0, atol=0)
    output.square().mean().backward()
    assert all(not p.requires_grad and p.grad is None for p in uniform.gate.parameters())
    for expert in uniform.experts:
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
    optimizer = Trainer.build_optimizer(config(), uniform)
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) not in optimized for p in uniform.gate.parameters())
    assert all(id(p) in optimized for expert in uniform.experts for p in expert.parameters())
    handle.remove()


def test_per_class_cad_matches_individual_shared_single_class_calls():
    shared = CostGuidanceRefiner(12, 8)
    per_class = ClassGuidanceRefiner(12, 8)
    per_class.load_state_dict(shared.state_dict(), strict=True)
    cost, guide = torch.randn(2, 3, 12, 8, 8), torch.randn(2, 8, 8, 8)
    actual = per_class(cost.reshape(6, 12, 8, 8), guide)
    expected = torch.cat([shared(cost[b, k:k + 1], guide[b:b + 1])
                          for b in range(2) for k in range(3)])
    torch.testing.assert_close(actual, expected)
    order = [2, 0, 1]
    reordered = per_class(cost[:, order].reshape(6, 12, 8, 8), guide).reshape(2, 3, 8, 8, 8)
    torch.testing.assert_close(reordered, actual.reshape(2, 3, 8, 8, 8)[:, order])


def test_per_class_refinement_gradient_does_not_cross_classes_or_images():
    refiner = ClassGuidanceRefiner(12, 8, residual_scale=0.5)
    cost = torch.randn(2, 3, 12, 8, 8, requires_grad=True)
    guide = torch.randn(2, 8, 8, 8, requires_grad=True)
    refined = refiner(cost.reshape(6, 12, 8, 8), guide).reshape(2, 3, 8, 8, 8)
    refined[0, 1].square().mean().backward()
    assert cost.grad[0, 1].abs().sum() > 0
    assert cost.grad[0, [0, 2]].abs().sum() == 0
    assert cost.grad[1].abs().sum() == 0
    assert guide.grad[0].abs().sum() > 0 and guide.grad[1].abs().sum() == 0


def test_per_class_cad_single_class_full_model_is_exactly_shared_cad():
    torch.manual_seed(42)
    shared = make_model(**FULL).eval()
    torch.manual_seed(42)
    per_class = make_model(**FULL, cad_scope='per_class').eval()
    arguments = inputs(batch=2, classes=1)
    with torch.no_grad():
        torch.testing.assert_close(per_class(*arguments), shared(*arguments), rtol=0, atol=0)


def test_per_class_decoder_receives_real_classes_and_both_stages_work():
    model = make_model(aggregation='parallel_epl', context_adapter='lsk',
                       cad_stages=(1, 2), cad_scope='per_class', pad_len=24)
    events = []
    def record(module, args, result):
        events.append((args[0].shape[0], args[1].shape[0], result.shape[0]))
    handles = [decoder.guidance_refiner.register_forward_hook(record)
               for decoder in (model.decoder1, model.decoder2)]
    assert model(*inputs(batch=2, classes=3)).shape == (2, 3, 32, 32)
    assert events == [(6, 2, 6), (6, 2, 6)]
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize('options', VARIANTS)
def test_new_prototypes_cpu_bfloat16_backward(options):
    model = make_model(**FULL, **options)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(*inputs(batch=2))
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('experiment,routing,scope', [
    ('full', 'learned', 'shared'),
    ('full_epl_uniform', 'uniform', 'shared'),
    ('full_class_cad', 'learned', 'per_class'),
    ('full_uniform_class_cad', 'uniform', 'per_class'),
])
def test_experiment_options_reach_predictor_and_checkpoint_contract(experiment, routing, scope):
    cfg = config(experiment)
    options = CATSegPredictor.from_config(cfg)
    assert options['epl_routing'] == routing and options['cad_scope'] == scope
    signature = model_config(cfg)
    assert signature.get('MODEL.OVRS.EPL_ROUTING', 'learned') == routing
    assert signature.get('MODEL.OVRS.CAD_SCOPE', 'shared') == scope
    if experiment == 'full':
        assert 'MODEL.OVRS.EPL_ROUTING' not in signature
        assert 'MODEL.OVRS.CAD_SCOPE' not in signature


@pytest.mark.parametrize('experiment', ['full_epl_uniform', 'full_class_cad'])
def test_equal_shaped_new_mechanism_checkpoint_is_rejected_before_loading(tmp_path, experiment):
    source, target = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    before = target.weight.detach().clone()
    OVRSCheckpointer(source, str(tmp_path), expected_config=model_config(config())).save('model')
    with pytest.raises(ValueError, match='configuration mismatch'):
        EvaluationCheckpointer(target, expected_config=model_config(config(experiment))).load(str(tmp_path / 'model.pth'))
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_inactive_cad_scope_does_not_change_baseline_checkpoint_contract():
    cfg = config('baseline')
    expected = model_config(cfg)
    cfg.MODEL.OVRS.CAD_SCOPE = 'per_class'
    assert model_config(cfg) == expected


@pytest.mark.parametrize('options,match', [
    ({'cad_scope': 'per_class', 'cad_pooling': 'weighted'}, 'CAD_POOLING mean'),
    ({'epl_routing': 'uniform'}, 'requires parallel_epl'),
    ({'epl_routing': 'invalid'}, 'learned/uniform'),
    ({'cad_scope': 'invalid'}, 'shared/per_class'),
])
def test_invalid_prototype_configuration_is_not_silently_ignored(options, match):
    with pytest.raises(ValueError, match=match):
        make_model(**options)


def test_decoder_explicit_class_count_rejects_wrong_guidance_layout():
    decoder = Up(32, 16, 8, cad=True, cad_scope='per_class')
    with pytest.raises(ValueError, match=r'B \* num_classes'):
        decoder(torch.randn(6, 32, 4, 4), torch.randn(6, 8, 8, 8), num_classes=3)
    with pytest.raises(ValueError, match='positive multiple'):
        decoder(torch.randn(5, 32, 4, 4), torch.randn(2, 8, 8, 8), num_classes=3)


@pytest.mark.parametrize('refiner_class', [CostGuidanceRefiner, ClassGuidanceRefiner])
def test_cad_rejects_empty_class_dimension(refiner_class):
    with pytest.raises(ValueError, match='positive integer multiple'):
        refiner_class(12, 8)(torch.empty(0, 12, 8, 8), torch.randn(2, 8, 8, 8))
