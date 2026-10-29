"""Seven matched cached-feature runs and open-vocabulary evaluation."""
import argparse, copy, hashlib, json, math, os, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
import open_clip
from data import load_cache, check_compatible, batch, sha256, write_json
from evaluate import evaluate
from model import SemanticStructureDecoder, optimization_step
from train_cached import state_digest, atomic_save

BASE=Path('/home/wangchengjun/huangziyi/semantic_structure_acceptance_20261028/model_v2_fulltrain_20261028')
MANIFEST=Path('/home/wangchengjun/huangziyi/ovrs_formal_20261011_dlrsd_v1/frozen_data/train_val/manifest.json')
CLIP=Path('/home/wangchengjun/huangziyi/open-ov/asset_store/backbones/clip/ViT-B-16.pt')
ARMS={'B':('mean','geometry',None),'C':('adaptive','geometry',None),
      'D':('mean','conditioned',None),'E':('adaptive','conditioned',None),
      'F':('mean','geometry',0),'G':('mean','geometry',1),'H':('mean','identity',None)}
CHECKPOINTS=(4000,10000,20000,40000,60000,80000)

def lr_at(step):
    if not 1 <= step <= 80000: raise ValueError('invalid step')
    if step <= 1000: return 2e-4*step/1000
    return 2e-6+(2e-4-2e-6)*0.5*(1+math.cos(math.pi*(step-1000)/79000))

def select(data, index):
    if index is None: return data
    x=dict(data); x['text']=data['text'][:,index:index+1].clone();x['valid']=data['valid'][:,index:index+1].clone()
    x['metadata']=copy.deepcopy(data['metadata'])
    x['metadata']['descriptions']=[[row[index]] for row in data['metadata']['descriptions']]
    x['metadata']['valid_mask']=x['valid'].tolist()
    assert x['valid'].all()
    return x

def prepare(out):
    old=json.loads((BASE/'protocol.json').read_text())
    assert sha256(MANIFEST)==old['manifest_sha256'] and sha256(CLIP)==old['encoder_weights_sha256']['clip']
    val=load_cache(BASE/'features/val.pt','validation'); info=json.loads(MANIFEST.read_text())
    names=json.loads((MANIFEST.parent/info['test_classes']).read_text()); seen=info['seen_class_ids']
    assert [names[i] for i in seen]==old['class_names']
    prompts=[[t.format(**{'class':name}) for t in old['templates']] for name in names]
    torch.set_num_threads(2)
    clip=open_clip.load_openai_model(str(CLIP),device='cpu',precision='fp32',jit=False).eval().requires_grad_(False)
    with torch.inference_mode():
        text=clip.encode_text(open_clip.get_tokenizer('ViT-B-16')([x for row in prompts for x in row])).cpu().reshape(len(names),2,-1)
    torch.testing.assert_close(text[seen],val['text'],rtol=1e-5,atol=1e-5)
    labels=[]
    for i,rec in enumerate(info['val']):
        oldrec=val['metadata']['records'][i]; path=MANIFEST.parent/rec['sem_seg_file_name']
        assert Path(rec['file_name']).stem==oldrec['id'] and sha256(path)==oldrec['mask_sha256']
        y=np.asarray(Image.open(path).resize((224,224),Image.Resampling.NEAREST),dtype=np.int64).copy()
        assert np.isin(y,list(range(17))+[255]).all()
        remap=np.full(y.shape,255,dtype=np.int64)
        for new,original in enumerate(seen):remap[y==original]=new
        assert torch.equal(torch.from_numpy(remap),val['labels'][i])
        labels.append(torch.from_numpy(y))
    result={'text':text,'valid':torch.ones(len(names),2,dtype=torch.bool),'labels':torch.stack(labels),
            'class_names':names,'descriptions':prompts,'seen_ids':seen,
            'unseen_ids':[i for i in range(len(names)) if i not in seen],
            'mask_sources':val['metadata']['records'],'clip_sha256':sha256(CLIP),
            'val_cache_sha256':sha256(BASE/'features/val.pt'),'manifest_sha256':sha256(MANIFEST)}
    torch.save(result,out/'validation17.pt')
    print('prepared 17 labels from raw val masks, no images extracted',flush=True)

def extended(val, extra, template):
    result=dict(val);result['text']=extra['text'];result['valid']=extra['valid'];result['labels']=extra['labels']
    result['metadata']=copy.deepcopy(val['metadata']);result['metadata']['class_names']=extra['class_names']
    result['metadata']['descriptions']=extra['descriptions'];result['metadata']['label_mapping']='original 17 IDs; original 255 retained'
    result['metadata']['preprocessing']['class_original_ids']=list(range(len(extra['class_names'])))
    return select(result,template)

def evaluate17(model,data,extra,device):
    result=evaluate(model,data,device,8)
    def average(ids):
        values=[result['iou'][i] for i in ids if result['iou'][i] is not None]
        return sum(values)/len(values) if values else None
    s=average(extra['seen_ids']);u=average(extra['unseen_ids'])
    result.update(seen_miou=s,unseen_miou=u,hiou=2*s*u/(s+u) if s is not None and u is not None and s+u else 0.,
                  seen_ids=extra['seen_ids'],unseen_ids=extra['unseen_ids'])
    return result

def train(root,arm,steps,resume=None,smoke=False):
    torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.cuda.init();torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats();device='cuda:0'
    semantic,spatial,template=ARMS[arm]
    train=load_cache(BASE/'features/train.pt','train');val=load_cache(BASE/'features/val.pt','validation')
    check_compatible(train,val)
    extra=torch.load(root/'validation17.pt',weights_only=True)
    assert extra['val_cache_sha256']==sha256(BASE/'features/val.pt') and extra['manifest_sha256']==sha256(MANIFEST)
    train=select(train,template);val=select(val,template);v17=extended(val,extra,template)
    # Extended vocabulary reuses exactly the audited validation images, structure, and preprocessing.
    assert v17['metadata']['sample_ids']==val['metadata']['sample_ids'] and v17['metadata']['image_sha256']==val['metadata']['image_sha256']
    assert v17['metadata']['encoders']==train['metadata']['encoders']
    a=copy.deepcopy(v17['metadata']['preprocessing']);b=copy.deepcopy(train['metadata']['preprocessing'])
    assert a.pop('class_original_ids')==list(range(17)) and b.pop('class_original_ids')==extra['seen_ids'] and a==b
    assert v17['visual'].shape[1]==train['visual'].shape[1] and [x.shape[1] for x in v17['structures']]==[x.shape[1] for x in train['structures']]
    eligible=(train['labels']!=255).flatten(1).any(1).nonzero().flatten()
    config=dict(clip_dim=train['visual'].shape[1],structure_dims=[x.shape[1] for x in train['structures']],width=64,semantic_mode=semantic,spatial_mode=spatial,temperature=.1,semantic_strength=1.)
    model=SemanticStructureDecoder(**config).to(device);initial=state_digest(model.state_dict())
    identities={'source':{n:sha256(Path(__file__).with_name(n)) for n in ('study_80k.py','model.py','data.py','evaluate.py','train_cached.py')},
                'train_cache':sha256(BASE/'features/train.pt'),'validation_cache':sha256(BASE/'features/val.pt'),
                'validation17':sha256(root/'validation17.pt')}
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);sampler=torch.Generator().manual_seed(43);updates=0;trace=[]
    if resume:
        c=torch.load(resume,map_location='cpu',weights_only=True);assert c['config']==config and c['arm']==arm and c['identities']==identities
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sampler.set_state(c['sampler_state']);torch.set_rng_state(c['torch_rng_state']);torch.cuda.set_rng_state_all(c['cuda_rng_state']);updates=c['updates'];trace=c['sample_trace'];initial=c['initial_sha']
    out=root/('smoke' if smoke else 'runs')/arm;out.mkdir(parents=True,exist_ok=False)
    write_json(out/'controls.json',{'arm':arm,'config':config,'seed':42,'sampler_seed':43,'batch':8,'steps':steps,'schedule':'linear step/1000 to 2e-4 then cosine to 2e-6 at 80000','initial_sha':initial,'base_commit':'945cc07d9f28cad45a1e33486297180e419456ad','parameters':sum(p.numel() for p in model.parameters()),'template_index':template,'identities':identities})
    start=time.perf_counter();model.train()
    with (out/'train.jsonl').open('w') as log:
        while updates<steps:
            indices=eligible[torch.randint(len(eligible),(8,),generator=sampler)]
            rate=lr_at(updates+1)
            for group in opt.param_groups:group['lr']=rate
            loss=optimization_step(model,opt,*batch(train,indices,device))
            if loss is None:raise RuntimeError('all-ignore sampled batch')
            updates+=1;trace.append(indices.tolist())
            log.write(json.dumps({'update':updates,'loss':loss,'lr':rate,'indices':indices.tolist()})+'\n')
            if updates%100==0:log.flush();print(arm,updates,loss,rate,flush=True)
            if updates in CHECKPOINTS or (smoke and updates==steps):
                ck={'arm':arm,'updates':updates,'config':config,'model':model.state_dict(),'optimizer':opt.state_dict(),'sampler_state':sampler.get_state(),'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'sample_trace':trace,'initial_sha':initial,'schedule':'linear1000_cosine80000_floor2e-6','identities':identities}
                atomic_save(ck,out/f'checkpoint_{updates:06d}.pt')
                r14=evaluate(model,val,device,8);r17=evaluate17(model,v17,extra,device)
                write_json(out/f'validation14_{updates:06d}.json',r14);write_json(out/f'validation17_{updates:06d}.json',r17)
    torch.cuda.synchronize();write_json(out/'summary.json',{'updates':updates,'initial_sha':initial,'sample_trace_sha256':hashlib.sha256(json.dumps(trace).encode()).hexdigest(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=list(ARMS));p.add_argument('--steps',type=int,default=80000);p.add_argument('--resume');p.add_argument('--smoke',action='store_true');a=p.parse_args();root=Path(a.root)
    if a.prepare:prepare(root)
    else:train(root,a.arm,a.steps,a.resume,a.smoke)
