"""Compare a complete CAT-Seg model to its bypassed RCT version on a real image."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from cat_seg import add_cat_seg_config
from research.ov_rival_bridge.install import install

manifest, weights, output = map(Path, sys.argv[1:])
data = json.loads(manifest.read_text())
cfg = get_cfg()
add_deeplab_config(cfg)
add_cat_seg_config(cfg)
cfg.merge_from_file('configs/ovrs/baseline.yaml')
cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(data['train_classes'])
cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(data['test_classes'])
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 35
model = build_model(cfg)
state = torch.load(weights, map_location='cpu', weights_only=False)
model.load_state_dict(state.get('model', state), strict=True)
model.eval()
image_path = data['val'][0]['file_name']
image = Image.open(image_path).convert('RGB').resize((384, 384))
tensor = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().cuda()
inputs = [{'image': tensor, 'height': 384, 'width': 384}]
torch.set_float32_matmul_precision('high')
with torch.inference_mode():
    native = model(inputs)[0]['sem_seg']
    install(model, 'R')
    model.sem_seg_head.predictor.transformer.rival_bypass = True
    bypass = model(inputs)[0]['sem_seg']
    diff = (native - bypass).abs()
    result = {'sample': image_path, 'classes': 35, 'precision': 'FP32; matmul high',
              'strict_checkpoint_load': True, 'max_abs_error': diff.max().item(),
              'mean_abs_error': diff.mean().item(), 'pass_at_1e-6': bool(diff.max().item() <= 1e-6)}
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
