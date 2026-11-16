import json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
root=Path('/home/wangchengjun/huangziyi/experiments/rct_v1_closeout_20261116')
plan=json.loads((root/'diagnostic_plan.json').read_text())
audit=json.loads((root/'checkpoint_manifest.json').read_text())
manifest=json.loads(Path(audit['A']['dataset']['path']).read_text())
rows=manifest['val'][:300]
names=json.loads((Path(audit['A']['dataset']['path']).parent/'classes.json').read_text())
assert [r['sample_id'] for r in rows]==plan['sample_ids']
def grid(arr,shape,hw,div,pad=255):
 h,w=map(int,hw);ph=(h+div-1)//div*div;pw=(w+div-1)//div*div
 t=torch.from_numpy(arr.astype(np.float32))[None,None]
 t=F.interpolate(t,size=(h,w),mode='nearest');t=F.pad(t,(0,pw-w,0,ph-h),value=pad)
 return F.interpolate(t,size=shape,mode='nearest')[0,0].numpy().astype(np.int64)
def getbase(r):
 z=np.load(root/'predictions/A'/f"{r['sample_id']}.npz")
 scores=z['scores'];truth=np.asarray(Image.open(r['sem_seg_file_name']),dtype=np.uint8)
 g=grid(truth,scores.shape[-2:],z['image_shape'],int(z['div']))
 return z,scores,truth,g
allm=[]
for r in rows:
 z,s,t,g=getbase(r);top=np.partition(s,-2,axis=0)[-2:]
 allm.append((top[-1]-top[-2])[g!=255])
median=float(np.median(np.concatenate(allm)))
(root/'frozen_threshold.json').write_text(json.dumps({'median':median,'source':'Base scores over valid 300-image coarse grid; fixed before reading comparisons'},indent=2))
pairs=[('M','V'),('V','R'),('A','R')]
per=[]; strata=[];trans=defaultdict(int);coverage=np.zeros(35,np.int64);presence=np.zeros(35,np.int64)
for n,r in enumerate(rows):
 z,s,t,g=getbase(r);valid=t!=255
 coverage+=np.bincount(t[valid],minlength=35)[:35];presence+=np.bincount(np.unique(t[valid]),minlength=35)[:35]>0
 preds={a:np.load(root/'predictions'/a/f"{r['sample_id']}.npz")['pred'] for a in 'AMVR'}
 assert all(p.shape==t.shape for p in preds.values())
 order=np.argsort(s,axis=0);winner=order[-1];runner=order[-2];margin=np.take_along_axis(s,order[-1:],axis=0)[0]-np.take_along_axis(s,order[-2:-1],axis=0)[0]
 gv=g!=255; switch=np.zeros_like(gv)
 H,W=gv.shape
 for dy in (-1,0,1):
  for dx in (-1,0,1):
   if dy==dx==0:continue
   y0=max(0,-dy);y1=min(H,H-dy);x0=max(0,-dx);x1=min(W,W-dx)
   a=(slice(y0,y1),slice(x0,x1));b=(slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))
   switch[a] |= gv[a]&gv[b]&(runner[a]!=runner[b])
 rank3=np.any(order[-3:]==g[None],axis=0)
 masks={'all':gv,'low_margin':gv&(margin<=median),'high_margin':gv&(margin>median),'switch':gv&switch,'stable':gv&~switch,'gt_top3':gv&rank3,'gt_not_top3':gv&~rank3}
 for cls in plan['strata']['classes']:masks['gt_'+cls]=gv&(g==names.index(cls))
 for old,new in pairs:
  a,b=preds[old],preds[new];ok0=a==t;ok1=b==t
  rec={'sample_id':r['sample_id'],'comparison':new+'-'+old,'valid':int(valid.sum()),'both_correct':int((valid&ok0&ok1).sum()),'corrected':int((valid&~ok0&ok1).sum()),'regressed':int((valid&ok0&~ok1).sum()),'both_wrong':int((valid&~ok0&~ok1).sum())}
  per.append(rec)
  change=valid&(a!=b)
  codes=t[change].astype(np.int32)*1225+a[change].astype(np.int32)*35+b[change].astype(np.int32)
  vals,cnt=np.unique(codes,return_counts=True)
  for v,c in zip(vals,cnt):trans[(new+'-'+old,int(v)//1225,(int(v)//35)%35,int(v)%35)]+=int(c)
  ga=grid(a,s.shape[-2:],z['image_shape'],int(z['div']));gb=grid(b,s.shape[-2:],z['image_shape'],int(z['div']))
  for label,mask in masks.items():
   c0=ga==g;c1=gb==g
   strata.append({'sample_id':r['sample_id'],'comparison':new+'-'+old,'stratum':label,'cells':int(mask.sum()),'corrected':int((mask&~c0&c1).sum()),'regressed':int((mask&c0&~c1).sum())})
 if (n+1)%50==0:print('analyzed',n+1,flush=True)
def write(name,rows):
 with (root/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('paired_per_image.csv',per);write('strata_per_image.csv',strata)
tr=[{'comparison':k[0],'gt':names[k[1]],'old_prediction':names[k[2]],'new_prediction':names[k[3]],'pixels':v} for k,v in sorted(trans.items(),key=lambda x:-x[1])]
write('error_transitions.csv',tr)
write('sample_class_coverage.csv',[{'class':names[i],'pixels':int(coverage[i]),'images':int(presence[i])} for i in range(35)])
rng=np.random.default_rng(42);boot=rng.integers(0,len(rows),size=(2000,len(rows)))
summary=[]
for comparison in [b+'-'+a for a,b in pairs]:
 for label in list(masks):
  rr=[x for x in strata if x['comparison']==comparison and x['stratum']==label]
  cell=np.array([x['cells'] for x in rr]);corr=np.array([x['corrected'] for x in rr]);reg=np.array([x['regressed'] for x in rr])
  nums=(corr-reg)[boot].sum(axis=1);den=cell[boot].sum(axis=1)
  vals=np.divide(nums,den,out=np.full(2000,np.nan),where=den>0)*100
  nz=cell>0; net=corr-reg
  summary.append({'comparison':comparison,'stratum':label,'cells':int(cell.sum()),'corrected':int(corr.sum()),'regressed':int(reg.sum()),'net_pp':float(100*net.sum()/cell.sum()) if cell.sum() else None,'images_with_cells':int(nz.sum()),'images_positive':int((net>0).sum()),'images_negative':int((net<0).sum()),'images_tied':int((nz&(net==0)).sum()),'bootstrap_low_pp':float(np.nanpercentile(vals,2.5)) if den.any() else None,'bootstrap_high_pp':float(np.nanpercentile(vals,97.5)) if den.any() else None})
write('paired_analysis.csv',summary)
print('DONE',median,flush=True)
