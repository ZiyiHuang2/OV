import csv,json
from pathlib import Path
import numpy as np
from PIL import Image
p=Path('/home/wangchengjun/huangziyi/experiments/rct_v1_closeout_20261116')
audit=json.loads((p/'checkpoint_manifest.json').read_text())
rows=json.loads(Path(audit['A']['dataset']['path']).read_text())['val'][:300]
names=json.loads((Path(audit['A']['dataset']['path']).parent/'classes.json').read_text())
pairs=[('M','V'),('V','R'),('A','R')]
data=[]
for row in rows:
 gt=np.asarray(Image.open(row['sem_seg_file_name']),dtype=np.uint8);valid=gt!=255
 pred={a:np.load(p/'predictions'/a/(row['sample_id']+'.npz'))['pred'] for a in 'AMVR'}
 for old,new in pairs:
  a=pred[old];b=pred[new];correct=valid&(a!=gt)&(b==gt);regress=valid&(a==gt)&(b!=gt)
  n=np.bincount(gt[valid],minlength=35);c=np.bincount(gt[correct],minlength=35);r=np.bincount(gt[regress],minlength=35)
  for i in range(35):
   data.append({'sample_id':row['sample_id'],'comparison':new+'-'+old,'class':names[i],'pixels':int(n[i]),'corrected':int(c[i]),'regressed':int(r[i])})
with (p/'paired_class_per_image.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
rng=np.random.default_rng(42);boot=rng.integers(0,300,size=(2000,300));summary=[]
for old,new in pairs:
 for name in names:
  rr=[x for x in data if x['comparison']==new+'-'+old and x['class']==name]
  pixels=np.array([x['pixels'] for x in rr]);c=np.array([x['corrected'] for x in rr]);r=np.array([x['regressed'] for x in rr]);delta=c-r
  denom=pixels[boot].sum(1);valid=denom>0
  vals=100*delta[boot].sum(1)[valid]/denom[valid]
  summary.append({'comparison':new+'-'+old,'class':name,'pixels':int(pixels.sum()),'images':int((pixels>0).sum()),'corrected':int(c.sum()),'regressed':int(r.sum()),'net_percentage_points':100*float(delta.sum())/pixels.sum() if pixels.sum() else '', 'images_positive':int((delta>0).sum()),'images_negative':int((delta<0).sum()),'bootstrap_low':float(np.percentile(vals,2.5)) if len(vals) else '', 'bootstrap_high':float(np.percentile(vals,97.5)) if len(vals) else '', 'nonempty_bootstraps':len(vals)})
with (p/'paired_class_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
per=list(csv.DictReader((p/'paired_per_image.csv').open()));out=[]
for old,new in pairs:
 rr=[r for r in per if r['comparison']==new+'-'+old]
 px=np.array([int(x['valid']) for x in rr]);c=np.array([int(x['corrected']) for x in rr]);d=np.array([int(x['regressed']) for x in rr]);net=c-d
 samples=100*net[boot].sum(1)/px[boot].sum(1)
 out.append({'comparison':new+'-'+old,'valid':int(px.sum()),'corrected':int(c.sum()),'regressed':int(d.sum()),'net_pp':float(100*net.sum()/px.sum()),'positive_images':int((net>0).sum()),'negative_images':int((net<0).sum()),'tied_images':int((net==0).sum()),'ci_low_pp':float(np.percentile(samples,2.5)),'ci_high_pp':float(np.percentile(samples,97.5))})
with (p/'paired_full_pixel_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(out)
