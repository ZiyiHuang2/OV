import math

import pytest
import torch
from torch import nn

from train_ovrs import EvaluationCheckpointer, harmonic_iou


def test_evaluation_loads_exact_checkpoint(tmp_path):
    source, destination = nn.Linear(3, 2), nn.Linear(3, 2)
    path = tmp_path / 'model.pth'
    torch.save({'model': source.state_dict()}, path)
    EvaluationCheckpointer(destination).load(str(path))
    torch.testing.assert_close(destination.weight, source.weight)


@pytest.mark.parametrize('mismatch', ['missing', 'unexpected', 'shape'])
def test_evaluation_rejects_wrong_architecture(tmp_path, mismatch):
    model = nn.Linear(3, 2)
    state = model.state_dict()
    if mismatch == 'missing':
        state.pop('bias')
    elif mismatch == 'unexpected':
        state['untrained_module'] = torch.zeros(1)
    else:
        state['weight'] = torch.zeros(2, 4)
    path = tmp_path / 'mismatch.pth'
    torch.save({'model': state}, path)
    with pytest.raises(ValueError, match='evaluation architecture'):
        EvaluationCheckpointer(model).load(str(path))


def test_harmonic_iou_handles_zero_and_absent_groups():
    assert harmonic_iou(0, 0) == 0
    assert harmonic_iou(30, 60) == 40
    assert math.isnan(harmonic_iou(float('nan'), 10))
