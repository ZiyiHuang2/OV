import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from cat_seg import add_cat_seg_config
from cat_seg.cat_seg_model import CATSeg
from cat_seg.engine.checkpoint import EvaluationCheckpointer, OVRSCheckpointer, model_config
from cat_seg.engine.records import dataset_summary, write_run_records
from cat_seg.modeling.postprocessing import postprocess_padded_prediction

ROOT = Path(__file__).resolve().parents[1]


def config(experiment='full'):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(str(ROOT / 'configs' / 'ovrs' / (experiment + '.yaml')))
    return cfg


def test_prediction_crops_in_padded_image_coordinates():
    grid = torch.linspace(0, 1, 96).repeat(2, 96, 1)
    result = postprocess_padded_prediction(grid, (640, 864), (640, 853), (320, 427))
    full_canvas = F.interpolate(grid[None], (640, 864), mode='bilinear', align_corners=False)
    expected = F.interpolate(full_canvas[:, :, :640, :853], (320, 427), mode='bilinear', align_corners=False)[0]
    old = F.interpolate(grid[None], (320, 427), mode='bilinear', align_corners=False)[0]
    torch.testing.assert_close(result, expected)
    assert (result - old).abs().max() > 0.008


@pytest.mark.parametrize('image_size', [(640, 640), (32, 64)])
def test_unpadded_prediction_uses_one_resize_without_low_res_crop(image_size):
    prediction = torch.randn(3, 96, 96)
    result = postprocess_padded_prediction(prediction, image_size, image_size, (70, 90))
    expected = F.interpolate(prediction[None], (70, 90), mode='bilinear', align_corners=False)[0]
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_normal_forward_returns_all_images_with_individual_geometry():
    # Exercise CATSeg.forward without downloading CLIP: fake only its encoder/head.
    model = CATSeg.__new__(CATSeg)
    nn.Module.__init__(model)
    model.register_buffer('pixel_mean', torch.zeros(3, 1, 1))
    model.register_buffer('clip_pixel_mean', torch.zeros(3, 1, 1))
    model.register_buffer('clip_pixel_std', torch.ones(3, 1, 1))
    model.size_divisibility = 32
    model.clip_resolution = (384, 384)
    model.sliding_window = False
    model.upsample1 = nn.Identity()
    model.upsample2 = nn.Identity()

    class Encoder(nn.Module):
        def encode_image(self, images, dense):
            batch = images.shape[0]
            feature = torch.zeros(batch, 577, 2)
            model.layers = [feature.permute(1, 0, 2)] * 2
            return feature

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = nn.Module()
            self.predictor.clip_model = Encoder()

        def forward(self, features, guidance):
            return torch.stack([torch.zeros(2, 96, 96), torch.ones(2, 96, 96)])

    model.sem_seg_head = Head()
    model.eval()
    records = [dict(image=torch.zeros(3, 63, 95), height=31, width=47),
               dict(image=torch.zeros(3, 64, 128), height=64, width=128)]
    with torch.no_grad():
        result = model(records)
    assert len(result) == 2
    assert result[0]['sem_seg'].shape == (2, 31, 47)
    assert result[1]['sem_seg'].shape == (2, 64, 128)
    torch.testing.assert_close(result[0]['sem_seg'], torch.full((2, 31, 47), 0.5))
    torch.testing.assert_close(result[1]['sem_seg'], torch.ones(2, 64, 128).sigmoid())


def test_dilation_mismatch_rejected_before_weights_are_loaded(tmp_path):
    source = nn.Conv2d(4, 4, 7, padding=9, dilation=3, groups=4)
    destination = nn.Conv2d(4, 4, 7, padding=3, dilation=1, groups=4)
    before = destination.weight.detach().clone()
    saved = model_config(config())
    OVRSCheckpointer(source, str(tmp_path), expected_config=saved).save('model')
    requested = copy.deepcopy(saved)
    requested['MODEL.OVRS.LSK_DILATION'] = 1
    with pytest.raises(ValueError, match='LSK_DILATION'):
        EvaluationCheckpointer(destination, expected_config=requested).load(str(tmp_path / 'model.pth'))
    torch.testing.assert_close(destination.weight, before)


def test_checkpoint_metadata_and_matching_configuration_load(tmp_path):
    signature = model_config(config('full_selective'))
    source, destination = nn.Linear(3, 2), nn.Linear(3, 2)
    OVRSCheckpointer(source, str(tmp_path), expected_config=signature, provenance={'run_id': 'test'}).save('model')
    saved = torch.load(tmp_path / 'model.pth', weights_only=False)
    assert saved['ovrs_metadata']['model_config'] == signature
    assert saved['ovrs_metadata']['provenance']['run_id'] == 'test'
    EvaluationCheckpointer(destination, expected_config=signature).load(str(tmp_path / 'model.pth'))
    torch.testing.assert_close(destination.weight, source.weight)


def test_legacy_configuration_requires_explicit_opt_in(tmp_path, caplog):
    source = nn.Linear(3, 2)
    torch.save({'model': source.state_dict()}, tmp_path / 'old.pth')
    signature = model_config(config())
    with pytest.raises(ValueError, match='no OVRS model configuration'):
        EvaluationCheckpointer(nn.Linear(3, 2), expected_config=signature).load(str(tmp_path / 'old.pth'))
    EvaluationCheckpointer(nn.Linear(3, 2), expected_config=signature,
                           allow_unverified=True).load(str(tmp_path / 'old.pth'))
    assert 'Unverified legacy' in caplog.text


def test_open_vocabulary_and_runtime_changes_do_not_change_model_contract():
    original = config()
    updated = original.clone()
    updated.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = '/different/train.json'
    updated.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = '/different/test.json'
    updated.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 40
    updated.MODEL.WEIGHTS = '/different/model.pth'
    updated.MODEL.DEVICE = 'cpu'
    updated.INPUT.MIN_SIZE_TEST = 320
    updated.MODEL.OVRS.LSK_SCALE = 0.9  # Learned value will be restored from weights.
    assert model_config(updated) == model_config(original)
    disabled = config('baseline')
    changed = disabled.clone()
    changed.MODEL.OVRS.LSK_DILATION = 1
    assert model_config(changed) == model_config(disabled)


@pytest.mark.parametrize('strict', [False, True])
def test_initialization_can_change_model_config_but_resume_cannot(tmp_path, strict):
    source = nn.Linear(3, 2)
    saved = model_config(config())
    OVRSCheckpointer(source, str(tmp_path), expected_config=saved).save('model')
    requested = copy.deepcopy(saved)
    requested['MODEL.OVRS.LSK_DILATION'] = 1
    loader = OVRSCheckpointer(nn.Linear(3, 2), expected_config=requested, strict=strict)
    if strict:
        with pytest.raises(ValueError, match='configuration mismatch'):
            loader.load(str(tmp_path / 'model.pth'))
    else:
        loader.load(str(tmp_path / 'model.pth'))


def manifest(tmp_path):
    path = tmp_path / 'data'
    path.mkdir()
    (path / 'train.json').write_text('["road"]')
    (path / 'test.json').write_text('["road", "river"]')
    (path / 'manifest.json').write_text(json.dumps({
        'train_classes': 'train.json', 'test_classes': 'test.json',
        'train': [{'file_name': 'a.jpg'}], 'val': [{'file_name': 'b.jpg'}],
        'seen_class_ids': [0], 'split_mode': 'explicit',
    }))
    return path / 'manifest.json'


def test_eval_and_resume_preserve_initial_training_records(tmp_path):
    cfg = config()
    cfg.OUTPUT_DIR = str(tmp_path / 'output')
    args = argparse.Namespace(dataset_manifest=str(manifest(tmp_path)), experiment='full',
                              eval_only=False, resume=False, allow_unverified_checkpoint=False)
    training = write_run_records(cfg, args, ROOT)
    output = Path(cfg.OUTPUT_DIR)
    snapshots = {name: (output / name).read_bytes() for name in ('config.yaml', 'run_provenance.json')}
    (output / 'last_checkpoint').write_text('model_final.pth')
    cfg.SOLVER.MAX_ITER += 100
    args.resume = True
    resumed = write_run_records(cfg, args, ROOT)
    args.resume, args.eval_only = False, True
    evaluation = write_run_records(cfg, args, ROOT)
    second_eval = write_run_records(cfg, args, ROOT)
    assert len({training, resumed, evaluation, second_eval}) == 4
    for name, content in snapshots.items():
        assert (output / name).read_bytes() == content
    for directory, mode in [(training, 'train'), (resumed, 'resume'), (evaluation, 'eval')]:
        record = json.loads((directory / 'run_provenance.json').read_text())
        assert record['mode'] == mode
        assert record['dataset']['manifest_sha256'] == hashlib.sha256(Path(args.dataset_manifest).read_bytes()).hexdigest()
        assert record['repository']['commit']
        assert (directory / 'config.yaml').exists()
    args.eval_only = False
    with pytest.raises(ValueError, match='already contains training records'):
        write_run_records(cfg, args, ROOT)
    for name, content in snapshots.items():
        assert (output / name).read_bytes() == content


def test_manifest_and_vocabulary_changes_are_recorded(tmp_path):
    path = manifest(tmp_path)
    initial = dataset_summary(path)
    (path.parent / 'test.json').write_text('["river", "road"]')
    assert dataset_summary(path)['vocabulary_sha256'] != initial['vocabulary_sha256']
    path.write_text(path.read_text() + '\n')
    assert dataset_summary(path)['manifest_sha256'] != initial['manifest_sha256']
