"""Matched C-series train/eval entry point; original run.py remains unchanged."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

# Existing entry point sets up the pinned local CLIP and DINO dependencies.
from run import ManifestSplit, PROMPT
from combination_model import CSeriesSegmenter
from combination_core import Config, segmentation_loss


class StatusSplit(ManifestSplit):
    def __getitem__(self, index):
        image, target, name = super().__getitem__(index)
        row = self.rows[index]
        status_path = row.get('pixel_status_file_name')
        if status_path:
            with Image.open(self.root / status_path) as im:
                status = torch.from_numpy(np.array(im, dtype=np.int64))
        else:
            status = (target != self.ignore_id).long()
        if status.shape != target.shape or not ((status >= 0) & (status <= 2)).all() or not torch.equal(status == 1, target != self.ignore_id):
            raise ValueError(f'Invalid annotation status for {name}')
        return image, target, status, name


def data_order(data, seed, epoch):
    return torch.randperm(len(data), generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def loader(data, args, order=None):
    return DataLoader(data, batch_size=args.batch_size, sampler=order, num_workers=args.workers,
                      pin_memory=torch.device(args.device).type == 'cuda')


def save(path, model, opt, meta, epoch, cursor, steps, seen):
    state = {'metadata': meta, 'core': model.core.state_dict(), 'optimizer': opt.state_dict(),
             'epoch': epoch, 'cursor': cursor, 'steps': steps, 'seen': seen,
             'rng_cpu': torch.get_rng_state(),
             'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def train(args, model, data, meta, state):
    model.train()
    text = model.text_features(data.classes)
    opt = torch.optim.AdamW(model.core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = start_cursor = steps = seen = 0
    if state:
        for key in ('weights', 'size', 'prompt', 'manifest', 'train_classes', 'config', 'seed', 'batch_size', 'lr', 'weight_decay', 'ignore_id', 'limit', 'alignment_weight', 'aux_weight'):
            if state['metadata'][key] != meta[key]:
                raise ValueError(f'Resume mismatch: {key}')
        model.core.load_state_dict(state['core'], strict=True)
        opt.load_state_dict(state['optimizer'])
        start_epoch, start_cursor, steps, seen = (state[k] for k in ('epoch', 'cursor', 'steps', 'seen'))
        torch.set_rng_state(state['rng_cpu'])
        if torch.cuda.is_available() and state['rng_cuda']:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
    if args.max_steps and steps >= args.max_steps:
        raise ValueError('Checkpoint already at requested max_steps')
    t = perf_counter(); audit = {}; total = 0.; updates = 0
    frozen = [(p, p._version) for name, p in model.named_parameters() if not name.startswith('core.')]
    for epoch in range(start_epoch, args.epochs):
        order = data_order(data, args.seed, epoch)
        cursor = start_cursor if epoch == start_epoch else 0
        for images, labels, status, ids in loader(data, args, order[cursor:]):
            cursor += len(images)
            if not (labels != data.ignore_id).any() and not (model.config.use_pixel and (status == 2).any()):
                continue
            images, labels, status = (x.to(args.device, non_blocking=True) for x in (images, labels, status))
            if args.audit and not audit:
                audit.update(model.audit_features(images[:1]))
            opt.zero_grad(set_to_none=True)
            r = model(images, text, labels=labels, ignore=data.ignore_id)
            loss = segmentation_loss(r['logits'], labels, data.ignore_id, model.config.use_pixel, status)
            if 'alignment_loss' in r:
                loss = loss + args.alignment_weight * r['alignment_loss']
            if 'aux_logits' in r:
                loss = loss + args.aux_weight * segmentation_loss(r['aux_logits'], labels, data.ignore_id)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss on {ids}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.core.parameters()):
                raise RuntimeError(f'Nonfinite gradient on {ids}')
            if args.audit and updates < 3:
                audit[f'grad_step_{updates + 1}'] = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in model.core.named_parameters()}
            opt.step()
            if model.config.use_rs:
                model.core.memory.update(r['regions'], r['region_ids'])
            steps += 1; seen += len(images); updates += 1; total += loss.item()
            if args.audit and updates == 1:
                audit['first_train_ids'] = list(ids)
            if steps % args.save_every == 0 or (args.max_steps and steps >= args.max_steps):
                save(args.output / 'state.pt', model, opt, meta, epoch, cursor, steps, seen)
            if steps == 1 or steps % 100 == 0:
                print(f'step={steps} epoch={epoch} loss={loss.item():.6f}', flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        if cursor == len(order):
            save(args.output / 'state.pt', model, opt, meta, epoch + 1, 0, steps, seen)
        if args.max_steps and steps >= args.max_steps:
            break
    if updates == 0:
        raise RuntimeError('No optimizer steps executed')
    assert all(p.grad is None and p._version == version for p, version in frozen), 'Frozen encoder mutated'
    return {**meta, 'steps': steps, 'seen_images': seen, 'mean_loss': total / updates,
            'wall_seconds': perf_counter() - t, 'audit': audit}


def mean_for(iou, indices):
    vals = [iou[i] for i in indices if iou[i] is not None]
    return sum(vals) / len(vals) if vals else None


@torch.no_grad()
def evaluate(args, model, data, meta):
    model.eval(); text = model.text_features(data.classes); k = len(data.classes)
    confusion = torch.zeros(k, k, dtype=torch.long, device=args.device)
    cuda = torch.device(args.device).type == 'cuda'
    if cuda:
        warm = torch.stack([data[i][0] for i in range(min(len(data), args.batch_size))]).to(args.device)
        for _ in range(args.warmup):
            model(warm, text, force_keep_all=args.force_keep_all)['logits'].argmax(1)
        torch.cuda.synchronize(); del warm
        torch.cuda.reset_peak_memory_stats()
    times, sizes = [], []; present_classes = retained_classes = retained_pixels = total_pixels = 0
    seen = set(model.train_classes); sample_digest = sha256(); peak = 0
    for images, labels, _, names in loader(data, args):
        images = images.to(args.device, non_blocking=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats(); start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
        else:
            start = perf_counter()
        r = model(images, text, force_keep_all=args.force_keep_all); pred = r['logits'].argmax(1)
        if cuda:
            end.record(); end.synchronize(); times.append(start.elapsed_time(end)); peak = max(peak, torch.cuda.max_memory_allocated())
        else:
            times.append((perf_counter() - start) * 1000)
        keep = r['kept']; sizes.extend([len(keep)] * len(images))
        labels = labels.to(args.device); valid = labels != data.ignore_id
        pairs = labels[valid] * k + pred[valid]
        confusion += torch.bincount(pairs, minlength=k*k).reshape(k,k)
        for y in labels:
            y = y[y != data.ignore_id]; unique = y.unique()
            present_classes += len(unique); retained_classes += torch.isin(unique, keep).sum().item()
            total_pixels += y.numel(); retained_pixels += torch.isin(y, keep).sum().item()
        for name in names: sample_digest.update((name+'\n').encode())
        del r, pred, images, labels, valid, pairs, y, unique, keep
    intersection = confusion.diag(); union = confusion.sum(0)+confusion.sum(1)-intersection
    iou = [100 * (intersection[i]/union[i]).item() if union[i] else None for i in range(k)]
    si = mean_for(iou, [i for i,c in enumerate(data.classes) if c in seen]); ui = mean_for(iou, [i for i,c in enumerate(data.classes) if c not in seen])
    harmonic = (2*si*ui/(si+ui) if si+ui else 0.) if si is not None and ui is not None else None
    return {**meta, 'test_classes': data.classes, 'miou_percent': mean_for(iou, range(k)),
            'seen_miou': si, 'unseen_miou': ui, 'harmonic_iou': harmonic,
            'class_iou': dict(zip(data.classes, iou)), 'confusion_matrix': confusion.tolist(),
            'sample_ids_sha256': sample_digest.hexdigest(), 'images': len(sizes),
            'mean_retained_classes': sum(sizes)/len(sizes), 'class_presence_recall': retained_classes/present_classes if present_classes else None,
            'pixel_recall': retained_pixels/total_pixels if total_pixels else None,
            'batch_ms_mean': sum(times)/len(times), 'batch_ms_p95': float(np.percentile(times,95)),
            'ms_per_image': sum(times)/len(sizes), 'peak_cuda_mib': peak/2**20 if cuda else None,
            'force_keep_all': args.force_keep_all}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train','eval'))
    p.add_argument('--variant', choices=('C0','B0','D0','C1','C2','C3','C4','C5','C6'))
    for name in ('manifest','clip-weights','dino-weights','output','checkpoint'):
        p.add_argument('--'+name, type=Path, required=name in ('manifest','clip-weights','output'))
    p.add_argument('--device', default='cuda'); p.add_argument('--size',type=int,default=384)
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--workers',type=int,default=0)
    p.add_argument('--limit',type=int); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=1); p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=.01); p.add_argument('--max-steps',type=int)
    p.add_argument('--save-every',type=int,default=100); p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--audit',action='store_true'); p.add_argument('--force-keep-all',action='store_true')
    p.add_argument('--affinity',choices=('visual','semantic','dual'),default='dual')
    p.add_argument('--fusion',choices=('learned','mean'),default='learned')
    p.add_argument('--neighbors',type=int,default=32); p.add_argument('--topk',type=int)
    p.add_argument('--rs',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--pixel',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--alignment-weight',type=float,default=.2); p.add_argument('--aux-weight',type=float,default=.2)
    args=p.parse_args()
    if args.batch_size<1 or args.save_every<1 or args.epochs<1 or args.alignment_weight<0 or args.aux_weight<0: p.error('Invalid training settings')
    if args.mode=='eval' and not args.checkpoint: p.error('Evaluation requires checkpoint')
    if args.mode=='train' and args.force_keep_all: p.error('Training always uses all categories')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True) if args.checkpoint else None
    if args.mode=='eval':
        cfg=Config(**state['metadata']['config']); train_classes=state['metadata']['train_classes']
        if args.variant and args.variant!=cfg.variant: p.error('Variant differs from checkpoint')
        if args.topk is not None:
            if cfg.variant!='C6': p.error('Top-k override requires C6')
            cfg=Config(**{**asdict(cfg),'topk':args.topk})
    else:
        if not args.variant: p.error('Training requires variant')
        cfg=Config(args.variant,args.affinity,args.neighbors,args.topk if args.topk is not None else 4,args.rs,args.pixel,args.fusion)
        train_classes=None
    split='train' if args.mode=='train' else 'val'; data=StatusSplit(args.manifest,split,args.limit)
    if not len(data) or len(set(data.classes))!=len(data.classes): p.error('Empty split or duplicate classes')
    if train_classes is None: train_classes=data.classes
    model=CSeriesSegmenter(args.clip_weights.expanduser(),args.dino_weights.expanduser() if args.dino_weights else None,args.size,cfg,train_classes).to(args.device)
    meta={'config':asdict(cfg),'weights':model.weight_hashes,'size':args.size,'prompt':PROMPT,
          'manifest':sha256(args.manifest.read_bytes()).hexdigest(),'train_classes':train_classes,
          'ignore_id':data.ignore_id,'seed':args.seed,'batch_size':args.batch_size,'lr':args.lr,
          'weight_decay':args.weight_decay,'alignment_weight':args.alignment_weight,'aux_weight':args.aux_weight,
          'limit':args.limit,'split':split,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'torch':str(torch.__version__),'device':str(args.device),'dtype':'float32','tf32':False}
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='train':
        if (args.output/'state.pt').exists() and not args.checkpoint: p.error('Refusing to overwrite a prior run; use a new directory or resume')
        report=train(args,model,data,meta,state)
    else:
        for key in ('weights','size','prompt'):
            if state['metadata'][key]!=meta[key]: raise ValueError(f'Eval mismatch: {key}')
        model.core.load_state_dict(state['core'],strict=True); report=evaluate(args,model,data,meta)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)

if __name__=='__main__': main()
