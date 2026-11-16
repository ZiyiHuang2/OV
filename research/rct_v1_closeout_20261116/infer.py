import os,sys,json
from pathlib import Path
import numpy as np, torch
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config
from train_ovrs import OVRSTrainer
out=Path('/home/wangchengjun/huangziyi/experiments/rct_v1_closeout_20261116')
plan=json.loads((out/'diagnostic_plan.json').read_text()); audit=json.loads((out/'checkpoint_manifest.json').read_text())
manifest=Path(audit['A']['dataset']['path'])
rows=json.loads(manifest.read_text())['val'][:300]
assert [r['sample_id'] for r in rows]==plan['sample_ids']
arm=sys.argv[1]; spec=audit[arm]
cfg=get_cfg();add_deeplab_config(cfg);add_cat_seg_config(cfg);cfg.merge_from_file(spec['config']);cfg.freeze()
torch.set_float32_matmul_precision('high')
model=OVRSTrainer.build_model(cfg)
model.load_state_dict(torch.load(spec['final_checkpoint'],map_location='cpu',weights_only=False)['model'],strict=True)
model.eval(); mapper=DatasetMapper(cfg,is_train=False)
capt={}
def hook(module,args,res): capt['scores']=res.detach().float().mean(dim=1).cpu().numpy()[0]
if arm=='A': model.sem_seg_head.predictor.transformer.register_forward_pre_hook(lambda m,a:None); model.sem_seg_head.predictor.transformer.correlation
# correlation is a method, not a module; intercept the bound method.
if arm=='A':
 agg=model.sem_seg_head.predictor.transformer; original=agg.correlation
 def wrapped(*a,**kw):
  res=original(*a,**kw);capt['scores']=res.detach().float().mean(dim=1).cpu().numpy()[0];return res
 agg.correlation=wrapped
dest=out/'predictions'/arm;dest.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():
 for i,row in enumerate(rows):
  path=dest/(row['sample_id']+'.npz')
  if path.exists(): raise RuntimeError('refuse overwrite '+str(path))
  item=mapper(dict(row)); pred=model([item])[0]['sem_seg'].argmax(0).cpu().numpy().astype(np.uint8)
  kw={'pred':pred}
  if arm=='A': kw.update(scores=capt['scores'],image_shape=np.array(item['image'].shape[-2:]),div=np.array(model.size_divisibility))
  np.savez_compressed(path,**kw)
  if (i+1)%50==0: print(arm,i+1,flush=True)
