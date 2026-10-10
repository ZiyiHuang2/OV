import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from tools.create_toy_dataset import create
from tools.prepare_rs_dataset import prepare, decode_mask
from cat_seg.data.datasets.register_rs import register_rs_manifest
from cat_seg.modeling.transformer.cat_seg_predictor import CATSegPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config
from train_ovrs import EXPERIMENTS


def test_prepare_seen_ids_and_register(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'prepared'
    create(source)
    args = argparse.Namespace(
        image_dir=source / 'images', mask_dir=source / 'labels', classes=source / 'classes.json',
        output=output, train_list=source / 'train.txt', val_list=source / 'val.txt',
        val_fraction=0.2, seed=42, seen_ids='2,0', label_map=None, color_map=None,
    )
    manifest = prepare(args)
    assert json.loads((output / 'classes_train.json').read_text()) == ['vegetation', 'building']
    train_mask = np.asarray(Image.open(output / manifest['train'][0]['sem_seg_file_name']))
    assert set(np.unique(train_mask)) == {0, 1, 255}
    val_mask = np.asarray(Image.open(output / manifest['val'][0]['sem_seg_file_name']))
    assert set(np.unique(val_mask)) == {0, 1, 2}
    registered = register_rs_manifest(output / 'manifest.json')
    assert len(DatasetCatalog.get(registered['train_name'])) == 6
    assert MetadataCatalog.get(registered['test_name']).seen_class_ids == [2, 0]
    assert registered['num_train_classes'] == 2


def test_rgb_masks_require_and_validate_color_mapping(tmp_path):
    path = tmp_path / 'rgb.png'
    pixels = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError, match='explicit'):
        decode_mask(path, 2)
    with pytest.raises(ValueError, match='unmapped'):
        decode_mask(path, 2, color_map={'255,0,0': 0})
    np.testing.assert_array_equal(decode_mask(path, 2, color_map={'255,0,0': 0, '0,255,0': 1}), [[0, 1]])


class DummyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 8)
        self.calls = 0

    def encode_text(self, tokens):
        self.calls += 1
        return self.embedding(tokens).mean(dim=1)


def test_text_cache_changes_with_vocab_mode_and_weights():
    predictor = CATSegPredictor.__new__(CATSegPredictor)
    nn.Module.__init__(predictor)
    predictor._token_cache = {}
    predictor._evaluation_cache = {}
    predictor.tokenizer = lambda texts: torch.tensor([[sum(text.encode()) % 256] for text in texts])
    predictor.clip_model = DummyCLIP()
    templates = ['{}', 'a {}']
    predictor.train()
    train_text = predictor.get_text_embeds(['road', 'building'], templates, predictor.clip_model)
    train_text.sum().backward()
    assert predictor.clip_model.embedding.weight.grad is not None
    predictor.eval()
    test_text = predictor.get_text_embeds(['river'], templates, predictor.clip_model)
    assert train_text.shape == (2, 2, 8) and test_text.shape == (1, 2, 8)
    calls = predictor.clip_model.calls
    predictor.get_text_embeds(['river'], templates, predictor.clip_model)
    assert predictor.clip_model.calls == calls
    predictor.load_state_dict(predictor.state_dict())
    predictor.get_text_embeds(['river'], templates, predictor.clip_model)
    assert predictor.clip_model.calls == calls + 1
    predictor.train()
    assert not predictor._evaluation_cache


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_experiment_config_merges(experiment):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    root = Path(__file__).resolve().parents[1]
    cfg.merge_from_file(str(root / 'configs' / 'ovrs' / (experiment + '.yaml')))
    assert cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS == 2
    assert cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE == 'attention'
