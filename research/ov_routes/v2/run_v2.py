"""Route B V2: strict source-only cache, selector training, and GT-free inference.

Run after copying research/ov_routes/v2/ into the *existing OV repository*:
 python -m research.ov_routes.v2.run_v2 --help

This does NOT modify original Route B files or the pretrained CAT-Seg weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from .engine import training_example, predict, first_pass
from .objective import selector_loss
from .selector import SpatialSemanticSelector, LegacySelectorExact, HeuristicSelector, IndexSelector
from .diagnostics import evaluate_cached_selector, ranking_pair_counts, summarize_target_alignment


SRC_FILES = ('evidence.py','selector.py','objective.py','engine.py','diagnostics.py','run_v2.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def source_identity():
    here = Path(__file__).resolve().parent
    return {p:sha(here/p) for p in SRC_FILES}


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def base_args(sub):
    sub.add_argument('--manifest',required=True)
    sub.add_argument('--config',required=True)
    sub.add_argument('--weights',required=True)
    sub.add_argument('--legacy-weights',action='store_true')
    sub.add_argument('--device',default='cuda')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser.add_subparsers(dest='task',required=True)
    cache=p.add_parser('precompute',help='SOURCE TRAIN only; never read val/test masks')
    base_args(cache)
    cache.add_argument('--cache-dir',required=True)
    cache.add_argument('--num-augmentations',type=int,default=2)
    cache.add_argument('--limit',type=int,default=0,help='SMOKE ONLY; restrict source train count')
    cache.add_argument('--sample-count',type=int,default=0,help='Formal reproducible random source subset; 0 means entire train split')
    cache.add_argument('--seed',type=int,default=42)
    cache.add_argument('--grid-size',type=int,default=8)
    cache.add_argument('--top-m',type=int,default=4)
    cache.add_argument('--with-raw-cost',action='store_true',help='Capture genuine CLIP correlation from CAT-Seg conv1 input')
    an=p.add_parser('analyze-cache',help='SOURCE cache BCE-vs-IoU target diagnostic; no network training')
    an.add_argument('--cache-dir',required=True)
    an.add_argument('--output',required=True)
    an.add_argument('--internal-val-fraction',type=float,default=0.10)
    an.add_argument('--split-seed',type=int,default=20261121)
    an.add_argument('--allow-smoke',action='store_true')
    tr=p.add_parser('train',help='Train only selector using precomputed source evidence')
    tr.add_argument('--cache-dir',required=True)
    tr.add_argument('--output',required=True)
    tr.add_argument('--steps',type=int,required=True)
    tr.add_argument('--save-every',type=int,default=1000)
    tr.add_argument('--batch-size',type=int,default=8)
    tr.add_argument('--lr',type=float,default=1e-4)
    tr.add_argument('--device',default='cuda')
    tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--hidden-dim',type=int,default=64)
    tr.add_argument('--ablation',choices=['legacy_exact','legacy_wide','legacy_only','spatial_only','spatial_cross',
                                           'full_mse','full_rank','cost_spatial','cost_full_rank'],default='spatial_only')
    tr.add_argument('--resume',help='Existing same-configuration V2 checkpoint')
    tr.add_argument('--internal-val-fraction',type=float,default=0.10,
                    help='Source-train original-image holdout; no external val GT used')
    tr.add_argument('--split-seed',type=int,default=20261121,
                    help='Fixed independent of training seed, so all ablations share same holdout')
    tr.add_argument('--allow-smoke',action='store_true',help='Allow training on intentionally truncated cache')
    ev=p.add_parser('evaluate',help='VAL labels only for metric; never selector choice')
    base_args(ev)
    ev.add_argument('--selector-checkpoint',help='Required only for learned policy')
    ev.add_argument('--policy',choices=['learned','zero','fixed','random','entropy','low_margin','disagreement'],default='learned')
    ev.add_argument('--output',required=True)
    ev.add_argument('--allow-skip',action='store_true',help='Prespecified predicted-gain <=0 no-op')
    ev.add_argument('--seed',type=int,default=42)
    return parser


def get_host(args):
    from ..data import Manifest
    from ..host import build_host
    data=Manifest(args.manifest)
    host,_=build_host(args.config,args.weights,data.train_classes,data.test_classes,
                      args.device,args.legacy_weights)
    host.requires_grad_(False)
    host.eval()
    return host,data


def dataset_files_sha256(data, split, indices=None):
    """Content hash, not merely path/manifest hash; detects silent file mutation.

    Split=train is used in the training cache identity. Validation file hashes
    are logged only at evaluation time, never used for selector supervision.
    """
    h = hashlib.sha256()
    unique_files = set()
    for idx in (range(len(data.rows[split])) if indices is None else indices):
        row=data.rows[split][idx]
        for field in ('file_name','sem_seg_file_name'):
            path=data.resolve(row[field])
            unique_files.add((str(path), str(row[field])))
    for path, relative in sorted(unique_files):
        digest=sha(path)
        h.update((relative+'\0'+digest+'\n').encode('utf-8'))
    return {'sha256':h.hexdigest(),'unique_files':len(unique_files)}


def base_model_source_sha256():
    """Track the V1 host/data and the actual CAT-Seg implementation in the repo.

    Real precompute/evaluation must run after this folder is copied into the
    original OV checkout; this review archive intentionally omits those files.
    """
    repo=Path(__file__).resolve().parents[3]
    core=repo/'cat_seg'
    v1=[repo/'research/ov_routes/data.py',repo/'research/ov_routes/host.py']
    if not core.is_dir() or any(not p.is_file() for p in v1):
        raise RuntimeError('Run inside the original OV checkout with CAT-Seg and V1 host/data present')
    files=sorted(v1+list(core.rglob('*.py')))
    h=hashlib.sha256()
    for path in files:
        rel=str(path.relative_to(repo))
        h.update((rel+'\0'+sha(path)+'\n').encode('utf-8'))
    return {'sha256':h.hexdigest(),'files':len(files)}


def core_identity(args,data,train_indices=None):
    return {'manifest_sha256':sha(args.manifest),
            'config_sha256':sha(args.config),
            'weights_sha256':sha(args.weights),
            'train_files':dataset_files_sha256(data,'train',train_indices),
            'host_and_catseg_source':base_model_source_sha256(),
            'train_names':list(data.names),
            'eval_names':list(data.all_names)}


def internal_split(samples, fraction, seed):
    """Group all augmentations of one original source image on the same side.

    If a manifest provides a parent_scene_id/scene_id, grouping follows that
    identifier. Otherwise the guarantee is source image identity, NOT geography.
    """
    if not 0.0 < fraction < 0.5:
        raise ValueError('Internal holdout fraction must be between 0 and 0.5')
    groups={}
    for idx,sample in enumerate(samples):
        meta=sample['meta']
        group=str(meta.get('source_group',meta.get('train_index',idx)))
        groups.setdefault(group,[]).append(idx)
    if len(groups)<2:
        raise ValueError('Need >=2 different original-image groups for internal holdout')
    keys=sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val=max(1,min(len(keys)-1,round(len(keys)*fraction)))
    val_groups=set(keys[:n_val])
    fit_idx=[i for key in keys if key not in val_groups for i in groups[key]]
    val_idx=[i for key in keys if key in val_groups for i in groups[key]]
    if not fit_idx or not val_idx:
        raise RuntimeError('Empty internal train/validation split')
    return fit_idx,val_idx,{'source_group_count':len(keys),
                            'fit_groups':sorted(set(keys)-val_groups),
                            'internal_val_groups':sorted(val_groups)}


def select_source_indices(total, count, seed):
    """Deterministic random source subset, no development-set information."""
    if total < 1 or count < 0:
        raise ValueError('Invalid source subset')
    if not count or count >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), count))


def verify_evaluation_identity(config, args, host, data):
    """Bind CAT-Seg checkpoint/config/code, not the source data manifest.

    An external dataset or vocabulary is allowed if the frozen model identity
    and selector text embedding width match the source training environment.
    No external GT is ever supplied to the selector.
    """
    source=config['cache_identity']
    current={'config_sha256':sha(args.config),'weights_sha256':sha(args.weights),
             'host_and_catseg_source':base_model_source_sha256()}
    for key,value in current.items():
        if source[key]!=value:
            raise ValueError('Evaluation CAT-Seg identity mismatch: '+key)
    embed_dim=host.text_vectors(data.all_names).shape[-1]
    if embed_dim!=config['settings']['text_dim']:
        raise ValueError('Selector text embedding width mismatch')
    return {'source_manifest_sha256':source['manifest_sha256'],
            'evaluation_manifest_sha256':sha(args.manifest),
            'cross_dataset':sha(args.manifest)!=source['manifest_sha256'],
            'source_class_count':len(source['train_names']),
            'evaluation_class_count':len(data.all_names)}


def do_precompute(a):
    from ..data import augment
    if a.num_augmentations<1 or a.num_augmentations>8 or a.limit<0 or a.sample_count<0:
        raise ValueError('Require 1..8 augmentations and nonnegative subset sizes')
    if a.limit and a.sample_count:
        raise ValueError('--limit is smoke-only; do not combine with formal --sample-count')
    seed_everything(a.seed)
    host,data=get_host(a)
    out=Path(a.cache_dir)
    out.mkdir(parents=True,exist_ok=False)
    selected=select_source_indices(len(data.rows['train']), a.sample_count, a.seed)
    if a.limit:
        selected=selected[:min(a.limit,len(selected))]
    image_count=len(selected)
    # Fail early if the candidate model is not faithful to the actual CAT-Seg checkpoint.
    first_image,first_labels=data.read('train',0)
    first_image,first_labels=augment(first_image,first_labels,torch.Generator().manual_seed(901))
    parity=host.check_native(first_image.to(a.device),data.names)
    meta={'format':'route-b-v2-cache-v2','source_only':True,'split':'train',
          'identity':core_identity(a,data,selected),'code':source_identity(),
          'selected_source_indices':selected,
          'n_images':image_count,'n_total_train':len(data.rows['train']),
          'num_augmentations':a.num_augmentations,'seed':a.seed,
          'grid_size':a.grid_size,'top_m':a.top_m,'with_raw_cost':a.with_raw_cost,
          'smoke_only':bool(a.limit),
          'formal_subset':bool(a.sample_count),
          'native_max_error':parity}
    if a.with_raw_cost:
        # Hook must be purely observational on the real full CAT-Seg forward.
        with torch.no_grad():
            a_logits,_=first_pass(host,first_image.to(a.device),data.names,with_raw_cost=False)
            b_logits,raw=first_pass(host,first_image.to(a.device),data.names,with_raw_cost=True)
        torch.testing.assert_close(a_logits,b_logits,rtol=0.,atol=1e-6)
        if raw is None or raw.shape[0] != len(data.names):
            raise RuntimeError('Raw CAT-Seg cost hook parity/shape preflight failed')
        meta['raw_cost_hook_max_abs_error']=float((a_logits-b_logits).abs().max())
    write_json(out/'meta.json',meta)
    count=0
    for i in selected:
        image,labels=data.read('train',i)
        for j in range(a.num_augmentations):
            rng=torch.Generator().manual_seed(a.seed+100003*i+719*j)
            rgb,gt=augment(image,labels,rng)
            sample=training_example(host,rgb.to(a.device),gt.to(a.device),data.names,
                                    grid_size=a.grid_size,top_m=a.top_m,
                                    with_raw_cost=a.with_raw_cost)
            if sample is None:
                continue
            row=data.rows['train'][i]
            group=row.get('parent_scene_id',row.get('scene_id',row.get('source_image_id',i)))
            sample['meta'].update(train_index=i,augmentation=j,source_group=str(group),
                                  file_name=row['file_name'])
            dest=out/f'train_{i:05d}_a{j:02d}.pt'
            torch.save(sample,dest)
            count+=1
    if not count:
        raise RuntimeError('No valid source-label samples in training cache')
    meta['examples_written']=count
    write_json(out/'meta.json',meta)
    print(json.dumps({'cache_examples':count,'native_max_error':parity}))


def model_settings(ablation,hidden_dim,text_dim,grid_size,top_m):
    if ablation=='legacy_wide':
        hidden_dim=120  # 62,761 params for D=512, vs spatial_only 62,850 (89 gap)
    use_spatial=ablation not in ('legacy_only','legacy_exact','legacy_wide')
    return dict(text_dim=text_dim,grid_size=grid_size,hidden_dim=hidden_dim,top_m=top_m,
                use_spatial=use_spatial,
                use_cross=ablation in ('spatial_cross','full_mse','full_rank','cost_full_rank'),
                use_action_attention=ablation in ('full_mse','full_rank','cost_full_rank'),
                use_raw_cost=ablation in ('cost_spatial','cost_full_rank'))


def create_selector(ablation,settings):
    if ablation in ('legacy_exact','legacy_wide'):
        return LegacySelectorExact(settings['text_dim'], settings['hidden_dim'])
    return SpatialSemanticSelector(**settings)


def do_analyze_cache(a):
    """No model or GT from held-out/external val is accessed."""
    cache=Path(a.cache_dir)
    meta=json.loads((cache/'meta.json').read_text())
    if meta['format']!='route-b-v2-cache-v2' or meta['split']!='train' or not meta['source_only']:
        raise ValueError('Refuse non-SOURCE TRAIN cache')
    if meta['code']!=source_identity():
        raise ValueError('Cache source identity mismatch')
    if meta['smoke_only'] and not a.allow_smoke:
        raise ValueError('Smoke cache requires explicit --allow-smoke')
    paths=sorted(cache.glob('train_*.pt'))
    if len(paths)!=meta['examples_written']:
        raise ValueError('Cache count mismatch')
    samples=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    fit_idx,val_idx,split=internal_split(samples,a.internal_val_fraction,a.split_seed)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'source_fit_target_alignment.json',summarize_target_alignment([samples[i] for i in fit_idx]))
    write_json(out/'source_internal_val_target_alignment.json',summarize_target_alignment([samples[i] for i in val_idx]))
    write_json(out/'split.json',{**split,'split_seed':a.split_seed,
                                 'internal_val_fraction':a.internal_val_fraction})
    print(json.dumps({'cache_examples':len(samples),'fit':len(fit_idx),
                      'internal_val':len(val_idx),'status':'posthoc_source_only'}))


def do_train(a):
    if a.steps<1 or a.batch_size<1 or a.lr<=0 or a.save_every<1:
        raise ValueError('Require positive steps, batch_size, lr, save_every')
    seed_everything(a.seed)
    cache=Path(a.cache_dir)
    meta=json.loads((cache/'meta.json').read_text())
    if meta['format']!='route-b-v2-cache-v2' or meta['split']!='train' or not meta['source_only']:
        raise ValueError('Refuse non-SOURCE TRAIN cache')
    if meta['code']!=source_identity():
        raise ValueError('Cache source identity mismatch; never mix code versions')
    if meta['smoke_only'] and not a.allow_smoke:
        raise ValueError('Truncated smoke cache cannot be used as formal training')
    paths=sorted(cache.glob('train_*.pt'))
    if len(paths)!=meta['examples_written']:
        raise ValueError('Cache count mismatch')
    samples=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    fit_idx,val_idx,split=internal_split(samples,a.internal_val_fraction,a.split_seed)
    dim=samples[0]['evidence']['class_tokens'].shape[-1]-2
    settings=model_settings(a.ablation,a.hidden_dim,dim,meta['grid_size'],meta['top_m'])
    if settings['use_raw_cost'] and not meta.get('with_raw_cost',False):
        raise ValueError('This ablation requires --with-raw-cost evidence cache')
    model=create_selector(a.ablation,settings).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    config={'cache_identity':meta['identity'],'cache_code':meta['code'],
            'source_code':source_identity(),'settings':settings,
            'ablation':a.ablation,'seed':a.seed,'steps':a.steps,
            'batch_size':a.batch_size,'lr':a.lr,
            'gain_scale':100.,'rank_weight':1. if a.ablation in ('full_rank','cost_full_rank') else 0.,
            'sample_count':len(samples),
            'fit_count':len(fit_idx),'internal_val_count':len(val_idx),
            'split_seed':a.split_seed,'internal_val_fraction':a.internal_val_fraction,
            'fit_indices_sha256':hashlib.sha256(json.dumps(fit_idx).encode()).hexdigest(),
            'val_indices_sha256':hashlib.sha256(json.dumps(val_idx).encode()).hexdigest()}
    generator=torch.Generator().manual_seed(a.seed+11)
    start=0
    if a.resume:
        state=torch.load(a.resume,map_location=a.device,weights_only=True)
        if config!=state['config']:
            raise ValueError('Cannot resume with different identity/settings')
        model.load_state_dict(state['selector'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sample_rng'])
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and 'cuda_rng' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=int(state['step'])
        if start>=a.steps:
            raise ValueError('Already at fixed endpoint')
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'protocol.json',config)
    write_json(out/'split.json',{**split,'fit_examples':len(fit_idx),
                                 'internal_val_examples':len(val_idx),
                                 'fixed_split_seed':a.split_seed})
    model.train()

    def save_checkpoint(path, step):
        checkpoint={'config':config,'selector':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'sample_rng':generator.get_state(),
                    'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'step':step}
        tmp=out/'checkpoint.tmp'
        torch.save(checkpoint,tmp)
        tmp.replace(path)

    with (out/'train.jsonl').open('w') as log:
        for step in range(start+1,a.steps+1):
            ids=torch.randint(len(fit_idx),(a.batch_size,),generator=generator)
            group=[samples[fit_idx[int(i)]] for i in ids]
            evidence={name:torch.stack([g['evidence'][name] for g in group])
                      for name in ('spatial','legacy','class_tokens','class_valid')}
            evidence={k:v.to(a.device) for k,v in evidence.items()}
            gains=torch.stack([g['gains'] for g in group]).to(a.device)
            pred=model(evidence)
            loss,detail=selector_loss(pred,gains,ranking_weight=config['rank_weight'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite selector loss at update {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step%100==0 or step==a.steps:
                pairs=ranking_pair_counts(gains)
                log.write(json.dumps({'step':step,'loss':float(loss.detach()),**detail,
                                      'rank_pairs':pairs})+'\n')
                log.flush()
                print(json.dumps({'step':step,'loss':float(loss.detach())}),flush=True)
            if step % a.save_every==0 and step!=a.steps:
                save_checkpoint(out/f'checkpoint_{step:06d}.pt',step)
    save_checkpoint(out/'selector_final.pt',a.steps)
    # Fixed endpoint: internal holdout is diagnostic only; no checkpoint selection.
    write_json(out/'internal_fit_diagnostics.json',evaluate_cached_selector(
        model,[samples[i] for i in fit_idx],a.device))
    write_json(out/'internal_val_diagnostics.json',evaluate_cached_selector(
        model,[samples[i] for i in val_idx],a.device))
    write_json(out/'COMPLETE.json',{'step':a.steps,'fixed_endpoint':True,
                                    'internal_val_used_for_checkpoint_selection':False,
                                    'external_val_used_for_training':False})


def do_evaluate(a):
    """External dev set evaluation; labels are used only for final metrics."""
    from ..data import metrics
    if a.policy != 'learned' and a.allow_skip:
        raise ValueError('No-op threshold applies only to calibrated learned BCE gains')
    if (a.policy=='learned') != bool(a.selector_checkpoint):
        raise ValueError('A selector checkpoint is required exactly for learned policy')
    seed_everything(a.seed)
    host,data=get_host(a)
    settings=None
    evaluation_identity=None
    if a.policy=='learned':
        c=torch.load(a.selector_checkpoint,map_location='cpu',weights_only=True)
        if not isinstance(c,dict) or 'config' not in c or 'selector' not in c:
            raise ValueError('Not a Route B V2 selector checkpoint')
        config=c['config']
        evaluation_identity=verify_evaluation_identity(config,a,host,data)
        if config['source_code']!=source_identity():
            raise ValueError('Selector model code changed after training')
        settings=config['settings']
        selector=create_selector(config['ablation'],settings).to(a.device)
        selector.load_state_dict(c['selector'],strict=True)
        selector.eval()
    elif a.policy in ('entropy','low_margin','disagreement'):
        selector=HeuristicSelector(a.policy).to(a.device)
    elif a.policy=='fixed':
        selector=IndexSelector(0).to(a.device)
    else:
        selector=None
    rng=torch.Generator().manual_seed(a.seed)
    n=len(data.all_names)
    cm=torch.zeros(n,n,dtype=torch.long)
    records=[]
    for i in range(len(data.rows['val'])):
        rgb,gt=data.read('val',i)
        if str(a.device).startswith('cuda'):
            torch.cuda.synchronize()
        t0=time.perf_counter()
        if a.policy=='zero':
            result=host.predict_logits(rgb[None].to(a.device),data.all_names)[0].sigmoid()
            trace={'selected':[],'encoder_calls':1,'skip':True,'boxes':[]}
        else:
            if a.policy=='random':
                selector=IndexSelector(int(torch.randperm(8,generator=rng)[0])).to(a.device)
            result,trace=predict(host,rgb.to(a.device),data.all_names,selector,
                                 grid_size=8 if settings is None else settings['grid_size'],
                                 top_m=4 if settings is None else settings['top_m'],
                                 allow_skip=a.allow_skip,scores_are_gain=(a.policy=='learned'))
        if str(a.device).startswith('cuda'):
            torch.cuda.synchronize()
        elapsed_ms=(time.perf_counter()-t0)*1000.
        pred=result.argmax(0).cpu()
        valid=gt!=255
        local=torch.bincount(gt[valid]*n+pred[valid],minlength=n*n).reshape(n,n)
        cm+=local
        records.append({'index':i,'file':data.rows['val'][i]['file_name'],
                        **trace,'wall_ms':elapsed_ms,**metrics(local,data.seen)})
    result={**metrics(cm,data.seen),'per_image':records,
            'model':'route-b-v2','policy':a.policy,
            'selector_checkpoint':str(a.selector_checkpoint) if a.selector_checkpoint else None,
            'evaluation_identity':evaluation_identity,
            'allow_skip':a.allow_skip,'validation_only':True,
            'oracle_or_gt_in_selection':False,'num_images':len(records),
            'val_data_files':dataset_files_sha256(data,'val'),
            'total_host_calls':sum(x['encoder_calls'] for x in records),
            'mean_wall_ms':statistics.mean(x['wall_ms'] for x in records),
            'median_wall_ms':statistics.median(x['wall_ms'] for x in records)}
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'validation.json',result)
    print(json.dumps({k:result[k] for k in ('miou','seen','unseen','hiou','total_host_calls')}))


def main():
    a=arguments().parse_args()
    if a.task=='precompute':do_precompute(a)
    elif a.task=='train':do_train(a)
    elif a.task=='analyze-cache':do_analyze_cache(a)
    elif a.task=='evaluate':do_evaluate(a)
    else:raise RuntimeError(a.task)


if __name__=='__main__':
    main()
