"""Paired full-model bypass and rival selection on fixed source-dev samples."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config
from train_ovrs import OVRSTrainer

config, checkpoint, manifest_path, out = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
rows = manifest['val'][:100]
cfg = get_cfg(); add_deeplab_config(cfg); add_cat_seg_config(cfg)
cfg.merge_from_file(str(config)); cfg.freeze()
torch.set_float32_matmul_precision('high')
model = OVRSTrainer.build_model(cfg)
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
model.eval()
aggregate = model.sem_seg_head.predictor.transformer
captured = {}
def capture(module, args):
    captured['scores'] = args[1].detach().cpu()
    captured['features'] = args[0].detach().float().square().mean().sqrt().item()
def update(module, args, result):
    captured['update'] = (result.detach().float() - args[0].detach().float()).square().mean().sqrt().item()
aggregate.rival_update.register_forward_pre_hook(capture)
aggregate.rival_update.register_forward_hook(update)
mapper = DatasetMapper(cfg, is_train=False)
counts = np.zeros(4, dtype=np.int64)
alternate_counts = np.zeros(4, dtype=np.int64)
conf = np.zeros((2, 35, 35), dtype=np.int64)
ratios = []
rival_match = rival_total = alternate_match = 0
original_rival = aggregate.rival_update._rival
def alternate_rival(scores, valid):
    _, margin, enough = original_rival(scores, valid)
    ranked = scores.float().masked_fill(~valid[:, :, None, None], float('-inf'))
    indices = ranked.topk(3, dim=1).indices
    ids = torch.arange(scores.shape[1], device=scores.device)[None, :, None, None]
    second = torch.where(indices[:, :1] == ids, indices[:, 2:3],
                         torch.where(indices[:, 1:2] == ids, indices[:, 2:3], indices[:, 1:2]))
    return second, margin, enough
with torch.inference_mode():
    for row in rows:
        item = mapper(dict(row))
        aggregate.rival_bypass = False
        active = model([item])[0]['sem_seg'].argmax(0).cpu().numpy()
        scores = captured['scores'][0]
        ratios.append(captured['update'] / max(captured['features'], 1e-12))
        aggregate.rival_update._rival = alternate_rival
        alternate = model([item])[0]['sem_seg'].argmax(0).cpu().numpy()
        aggregate.rival_update._rival = original_rival
        aggregate.rival_bypass = True
        bypass = model([item])[0]['sem_seg'].argmax(0).cpu().numpy()
        target = np.asarray(Image.open(row['sem_seg_file_name']), dtype=np.int64)
        valid = target != 255
        ok0 = bypass == target; ok1 = active == target
        counts += [np.count_nonzero(valid & ok0 & ok1), np.count_nonzero(valid & ~ok0 & ok1),
                   np.count_nonzero(valid & ok0 & ~ok1), np.count_nonzero(valid & ~ok0 & ~ok1)]
        alt_ok = alternate == target
        alternate_counts += [np.count_nonzero(valid & ok1 & alt_ok), np.count_nonzero(valid & ~ok1 & alt_ok),
                             np.count_nonzero(valid & ok1 & ~alt_ok), np.count_nonzero(valid & ~ok1 & ~alt_ok)]
        for idx, pred in enumerate((bypass, active)):
            conf[idx] += np.bincount(35 * target[valid] + pred[valid], minlength=35*35).reshape(35, 35)
        # Map target/prediction to the exact pre-aggregation grid, including input padding.
        h, w = item['image'].shape[-2:]; div = model.size_divisibility
        ph = ((h + div - 1) // div) * div; pw = ((w + div - 1) // div) * div
        def grid(array):
            x = torch.from_numpy(array.astype(np.float32))[None, None]
            x = F.interpolate(x, size=(h, w), mode='nearest')
            x = F.pad(x, (0, pw-w, 0, ph-h), value=255)
            return F.interpolate(x, size=scores.shape[-2:], mode='nearest')[0, 0].long()
        gt, pred = grid(target), grid(active)
        top = scores.topk(3, dim=0).indices
        rival = torch.where(top[0] == pred, top[1], top[0])
        alternate_id = torch.where(top[0] == pred, top[2], torch.where(top[1] == pred, top[2], top[1]))
        mistakes = (gt != 255) & (gt != pred)
        rival_total += int(mistakes.sum())
        rival_match += int((mistakes & (rival == gt)).sum())
        alternate_match += int((mistakes & (alternate_id == gt)).sum())
def miou(c):
    tp = c.diagonal(); gt = c.sum(1); pred = c.sum(0); union = gt + pred - tp
    valid = gt > 0
    return float((tp[valid] / union[valid]).mean() * 100)
result = {'scope': 'First 100 fixed source-dev images; diagnostic only, not independent test',
          'checkpoint': str(checkpoint), 'arm': cfg.MODEL.OVRS.RIVAL_ARM, 'images': len(rows),
          'bypass_miou': miou(conf[0]), 'enabled_miou': miou(conf[1]),
          'pixel_counts': dict(zip(['both_correct', 'corrected', 'regressed', 'both_wrong'], map(int, counts))),
          'alternative_rival_pixel_counts': dict(zip(['both_correct', 'corrected', 'regressed', 'both_wrong'], map(int, alternate_counts))),
          'mean_update_to_host_rms': float(np.mean(ratios)),
          'rival_of_predicted_class_matches_ground_truth_on_error_grid': {'matches': rival_match, 'alternative_matches': alternate_match, 'errors': rival_total},
          'note': 'Alternative intervention swaps to the second rival while holding the original margin fixed; R also changes the same-rival indicator.'}
out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
