import pytest
import torch
from torch import nn
from .bridge import RivalConditionedTransfer
from .install import install


class FakeLayer(nn.Module):
    def __init__(self, aggregation='serial', cost_context=None):
        super().__init__()
        self.aggregation = aggregation
        self.cost_context = cost_context


class FakeAgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 12
        self.layers = nn.ModuleList([FakeLayer()])


class FakePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = FakeAgg()


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = FakePredictor()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sem_seg_head = FakeHead()
        self.parameter = nn.Parameter(torch.ones(1))


@pytest.mark.parametrize('arm', ['S', 'M', 'R', 'P'])
def test_installer_and_optimizer(arm):
    model = FakeModel()
    state_before = set(model.state_dict().keys())
    result = install(model, arm, rank=5, attn_dim=3)
    assert result is model
    assert isinstance(model.sem_seg_head.predictor.transformer.rival_update, RivalConditionedTransfer)
    updated_names = set(model.state_dict().keys()) - state_before
    assert updated_names and all('.rival_update.' in n for n in updated_names)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    bridge = model.sem_seg_head.predictor.transformer.rival_update
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert {id(p) for p in bridge.parameters()} <= parameter_ids
    with pytest.raises(ValueError, match='already installed'):
        install(model, arm)


def test_reject_unsupported_host():
    model = FakeModel()
    model.sem_seg_head.predictor.transformer.layers[0].aggregation = 'parallel_simple'
    with pytest.raises(ValueError):
        install(model, 'R')


def test_reject_competition_module():
    model = FakeModel()
    model.sem_seg_head.predictor.transformer.competition_update = nn.Identity()
    with pytest.raises(ValueError):
        install(model, 'R')


def test_base_state_and_module_initialization_strict():
    model = FakeModel()
    before = {k: t.clone() for k, t in model.state_dict().items()}
    install(model, 'R', rank=5)
    assert all(torch.equal(model.state_dict()[k], v) for k,v in before.items())
    with pytest.raises(RuntimeError):
        model.load_state_dict(before, strict=True)
