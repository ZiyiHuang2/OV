"""CPU algorithm tests; synthetic models are NEVER substituted by the real runner."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from .episodic import masked_bce, objective, remap_support
from .views import action_features, choose, fuse, predict_views, training_actions, windows, ViewSelector
from .data import Manifest, metrics
from .run import restore, state

torch.set_num_threads(1)


class ToyHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.2, -.4, .7]))
        self.clip = nn.Linear(1, 1).requires_grad_(False)
        self.calls = []

    def text_vectors(self, names):
        return torch.tensor([[float(ord(n[0]) % 7)/7, .2, float(len(n))/10] for n in names],
                            device=self.weight.device)

    def forward(self, image, names):
        self.calls.append((tuple(image.shape), tuple(names)))
        x = image/255
        # A shared semantic mapping whose predictions change when actual views change.
        x = x + x.mean((-2, -1), keepdim=True)*.3
        return torch.einsum('bchw,kc,c->bkhw', x, self.text_vectors(names), self.weight)

    def predict_logits(self, image, names):
        return self(image, names)

    def check_native(self, image, names):
        return 0.  # Mock ONLY: runner integration test, not native CAT-Seg acceptance.


def inputs():
    torch.manual_seed(8)
    return torch.rand(1, 3, 5, 7)*255, torch.randint(3, (1, 5, 7))


def test_first_order_gradient_matches_explicit_formula():
    m = ToyHost()
    x, y = inputs(); q = x.flip(-1); names = ['soil', 'grass', 'water']
    before = state(m)
    loss, _ = objective(m, x, y, q, y, names, [2], inner_lr=.1)
    actual = torch.autograd.grad(loss, m.weight)[0]
    support = masked_bce(m(x, names[:2]), remap_support(y, [0, 1]))
    gs = torch.autograd.grad(support, m.weight)[0]
    fast = copy.deepcopy(m)
    with torch.no_grad(): fast.weight.copy_(m.weight-.1*gs)
    query = masked_bce(fast(q, names), y, [2])
    gq = torch.autograd.grad(query, fast.weight)[0]
    torch.testing.assert_close(actual, gs+gq)
    torch.testing.assert_close(m.weight, before['weight'])
    assert not torch.allclose(gq, torch.zeros_like(gq))


def test_episode_uses_subset_inner_full_outer():
    m = ToyHost(); x, y = inputs()
    loss, _ = objective(m, x, y, x, y, ['soil', 'grass', 'water'], [1])
    assert m.calls[0][1] == ('soil', 'water')
    assert m.calls[1][1] == ('soil', 'grass', 'water')
    assert loss.requires_grad


def test_held_support_pixels_have_zero_direct_gradient():
    x = torch.randn(1, 2, 2, 2, requires_grad=True)
    labels = torch.tensor([[[0, 1], [2, 255]]])
    loss = masked_bce(x, remap_support(labels, [0, 1]))
    loss.backward()
    assert x.grad[:, :, 1, :].count_nonzero() == 0


def test_all_ignore_skips_objective():
    m = ToyHost(); x, y = inputs(); y.fill_(255)
    before = state(m)
    loss, _ = objective(m, x, y, x, y, ['soil', 'grass', 'water'], [1])
    assert loss is None and m.weight.grad is None
    torch.testing.assert_close(m.weight, before['weight'])


def test_joint_is_distinct_from_meta():
    m = ToyHost(); x, y = inputs(); names = ['soil', 'grass', 'water']
    a, _ = objective(m, x, y, x, y, names, [1], inner_lr=.8)
    b, _ = objective(m, x, y, x, y, names, [1], inner_lr=.8, mode='joint')
    assert not torch.allclose(a, b)


def test_action_features_are_class_permutation_invariant():
    torch.manual_seed(3)
    z = torch.randn(4, 9, 13); t = torch.randn(4, 6); p = torch.tensor([2, 0, 3, 1])
    a = action_features(z, t, windows(9, 13))
    b = action_features(z[p], t[p], windows(9, 13))
    torch.testing.assert_close(a, b)


def test_zero_budget_and_true_crop_calls():
    m = ToyHost(); x, _ = inputs(); names = ['soil', 'grass', 'water']
    base = m(x, names)[0].sigmoid(); m.calls.clear()
    out, trace = predict_views(m, x[0], names, budget=0)
    torch.testing.assert_close(out, base)
    assert len(m.calls) == 1 and trace['encoder_calls'] == 1
    m.calls.clear()
    _, trace = predict_views(m, x[0], names, budget=2)
    assert len(m.calls) == 3 and trace['encoder_calls'] == 3
    assert all(c[0][-2:] != x.shape[-2:] for c in m.calls[1:])


def test_overlap_fusion_and_uncovered_pixels():
    base = torch.zeros(2, 6, 8)
    a = ((0, 0, 3, 4), torch.ones(2, 3, 4))
    b = ((1, 2, 5, 6), -torch.ones(2, 4, 4))
    ab = fuse(base, [a, b]); ba = fuse(base, [b, a])
    torch.testing.assert_close(ab, ba)
    torch.testing.assert_close(ab[:, 5], base.sigmoid()[:, 5])


def test_action_training_no_validation_labels_or_host_gradients():
    m = ToyHost(); x, y = inputs()
    features, gains = training_actions(m, x[0], y[0], ['soil', 'grass', 'water'])
    assert features.shape[0] == 8 and gains.shape == (8,)
    assert not features.requires_grad and not gains.requires_grad
    selector = ViewSelector(3)
    ((selector(features)-100*gains)**2).mean().backward()
    assert m.weight.grad is None
    assert sum(p.grad.abs().sum() for p in selector.parameters()) > 0
    assert training_actions(m, x[0], y[0].new_full(y[0].shape, 255), ['a', 'b', 'c']) is None


def test_policies_equal_budget_and_reproducible():
    features = torch.randn(8, 12); selector = ViewSelector(3)
    for policy in ('fixed', 'random', 'learned'):
        a = choose(features, 2, policy, torch.Generator().manual_seed(3), selector)
        b = choose(features, 2, policy, torch.Generator().manual_seed(3), selector)
        assert a == b and len(set(a)) == 2
    with pytest.raises(ValueError): choose(features, 9, 'fixed')


def test_confusion_full_competition_metrics():
    cm = torch.tensor([[8, 1, 1], [0, 5, 5], [2, 3, 5]])
    r = metrics(cm, [0, 1])
    assert r['unseen'] == pytest.approx(5/16)
    assert r['hiou'] == pytest.approx(2*r['seen']*r['unseen']/(r['seen']+r['unseen']))


def make_manifest(tmp):
    rows = {'train': [], 'val': []}
    for split in rows:
        for i in range(2):
            name = f'{split}{i}'
            image = np.random.default_rng(i).integers(0, 255, (9, 13, 3), dtype=np.uint8)
            mask = np.indices((9, 13)).sum(0).astype(np.uint8) % (2 if split == 'train' else 3)
            Image.fromarray(image).save(tmp/f'{name}.png')
            Image.fromarray(mask).save(tmp/f'{name}_mask.png')
            rows[split].append({'file_name': f'{name}.png', 'sem_seg_file_name': f'{name}_mask.png'})
    (tmp/'train.json').write_text(json.dumps(['soil', 'water']))
    (tmp/'classes.json').write_text(json.dumps(['soil', 'airplane', 'water']))
    info = {'version': 1, 'ignore_label': 255, 'train_classes': 'train.json',
            'test_classes': 'classes.json', 'seen_class_ids': [0, 2],
            'test': [{'file_name': '/MUST_NOT_OPEN_TEST', 'sem_seg_file_name': '/MUST_NOT_OPEN_TEST'}], **rows}
    path = tmp/'manifest.json'; path.write_text(json.dumps(info)); return path


def test_manifest_contract_and_test_sealing(tmp_path):
    data = Manifest(make_manifest(tmp_path))
    assert data.support() == [{0, 1}, {0, 1}]
    assert data.read('val', 0)[1].max() == 2
    with pytest.raises(ValueError): data.read('test', 0)


def test_real_file_runner_with_mock_host_train_reload_and_resume(tmp_path, monkeypatch):
    """End-to-end I/O/optimizer exercise, explicitly NOT a pretrained-model test."""
    from . import run
    manifest = make_manifest(tmp_path)
    weights = tmp_path/'weights.pth'; weights.write_text('mock host only')
    monkeypatch.setattr(run, 'build_host', lambda *args: (ToyHost(), 'mock config'))
    def launch(task, folder, *extra):
        args = ['run', task, '--manifest', str(manifest), '--config', 'mock',
                '--weights', str(weights), '--device', 'cpu', '--output', str(tmp_path/folder),
                '--eval-every', '1', *extra]
        monkeypatch.setattr(sys, 'argv', args); run.main()
    for task in ('train-a', 'train-b'):
        launch(task, task, '--steps', '2')
        directory = tmp_path/task
        checkpoint = directory/'checkpoint_000002.pt'
        saved = torch.load(checkpoint, weights_only=True)
        assert saved['updates'] == 2 and (directory/'COMPLETE.json').exists()
        launch(task, task+'-resume', '--steps', '2', '--resume', str(directory/'checkpoint_000001.pt'))
        resumed = torch.load(tmp_path/(task+'-resume')/'checkpoint_000002.pt', weights_only=True)
        for name, value in saved['parameters'].items():
            torch.testing.assert_close(value, resumed['parameters'][name], rtol=0, atol=0)
        for key in ('rng', 'aug_rng', 'torch_rng'):
            torch.testing.assert_close(saved[key], resumed[key], rtol=0, atol=0)
        for parameter_id, slots in saved['optimizer']['state'].items():
            for key, value in slots.items():
                torch.testing.assert_close(value, resumed['optimizer']['state'][parameter_id][key], rtol=0, atol=0)
        launch('evaluate', task+'-eval', '--artifact', str(checkpoint),
               '--policy', 'learned' if task == 'train-b' else 'fixed', '--budget', '1')
        assert (tmp_path/(task+'-eval')/'validation.json').exists()


def test_state_restore_rejects_partial_checkpoint():
    m = ToyHost()
    with pytest.raises(ValueError): restore(m, {})
