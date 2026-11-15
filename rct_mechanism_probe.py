"""Post-hoc coarse-grid diagnostic; the first 100 source-dev examples are fixed."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
import torch
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config
from train_ovrs import OVRSTrainer

config, checkpoint, manifest_path, old_pred, new_pred, output = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
rows = manifest['val'][:100]
sample_ids = {row['file_name'] for row in rows}
def load_predictions(path):
    groups = defaultdict(list)
    for item in json.loads(path.read_text()):
        if item['file_name'] in sample_ids:
            groups[item['file_name']].append(item)
    assert set(groups) == sample_ids
    return groups
old, new = load_predictions(old_pred), load_predictions(new_pred)
def decode(items, shape):
    result = np.full(shape, -1, np.int16)
    for item in items:
        mask = mask_util.decode(item['segmentation']).astype(bool)
        assert not np.any(result[mask] != -1)
        result[mask] = item['category_id']
    assert np.all(result >= 0)
    return torch.from_numpy(result.astype(np.float32))[None, None]

cfg = get_cfg(); add_deeplab_config(cfg); add_cat_seg_config(cfg)
cfg.merge_from_file(str(config)); cfg.freeze()
model = OVRSTrainer.build_model(cfg)
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
model.eval(); torch.set_float32_matmul_precision('high')
capture = {}
def hook(module, args):
    capture['scores'] = args[1].detach().cpu()
model.sem_seg_head.predictor.transformer.rival_update.register_forward_pre_hook(hook)
mapper = DatasetMapper(cfg, is_train=False)
chunks = []
with torch.inference_mode():
    for row in rows:
        item = mapper(dict(row))
        model([item])
        scores = capture['scores'][0]
        assert scores.shape[0] == len(manifest['seen_class_ids']) == 35
        h, w = item['image'].shape[-2:]
        div = model.size_divisibility
        ph, pw = ((h + div - 1)//div)*div, ((w + div - 1)//div)*div
        def grid(tensor, pad):
            tensor = F.interpolate(tensor, size=(h, w), mode='nearest')
            tensor = F.pad(tensor, (0, pw-w, 0, ph-h), value=pad)
            return F.interpolate(tensor, size=scores.shape[-2:], mode='nearest')[0, 0].long()
        truth_np = np.asarray(Image.open(row['sem_seg_file_name']), dtype=np.int16)
        truth = grid(torch.from_numpy(truth_np.astype(np.float32))[None, None], 255)
        a = grid(decode(old[row['file_name']], truth_np.shape), 255)
        b = grid(decode(new[row['file_name']], truth_np.shape), 255)
        valid = truth != 255
        topval, topid = scores.topk(2, dim=0)
        rival = torch.where(topid[0] != truth, topid[0], topid[1])
        safe_truth = truth.clamp(0, 34)
        class_score = scores.gather(0, safe_truth[None])[0]
        rival_score = scores.gather(0, rival[None])[0]
        margin = (class_score-rival_score).abs()
        switch = torch.zeros_like(valid)
        horizontal = valid[:, 1:] & valid[:, :-1] & (truth[:, 1:] == truth[:, :-1]) & (rival[:, 1:] != rival[:, :-1])
        vertical = valid[1:, :] & valid[:-1, :] & (truth[1:, :] == truth[:-1, :]) & (rival[1:, :] != rival[:-1, :])
        switch[:, 1:] |= horizontal; switch[:, :-1] |= horizontal
        switch[1:, :] |= vertical; switch[:-1, :] |= vertical
        chunks.append(torch.stack([margin[valid], switch[valid].float(), (a[valid] == truth[valid]).float(), (b[valid] == truth[valid]).float()], dim=1))
data = torch.cat(chunks).numpy()
median = float(np.median(data[:, 0]))
def summary(mask):
    part = data[mask]; before = part[:, 2].astype(bool); after = part[:, 3].astype(bool)
    return {'cells': int(len(part)), 'corrected': int((~before & after).sum()), 'regressed': int((before & ~after).sum())}
result = {'scope': 'First 100 source-dev images, coarse native feature grid; descriptive post-hoc diagnostic',
          'margin': 'absolute ground-truth-class score minus its strongest rival score; median split',
          'margin_median': median, 'high_competition': summary(data[:, 0] <= median),
          'low_competition': summary(data[:, 0] > median),
          'rival_switch': 'at least one adjacent valid same-GT-class cell has a different GT-class rival',
          'switch': summary(data[:, 1] > 0), 'no_switch': summary(data[:, 1] == 0)}
output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
