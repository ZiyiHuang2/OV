"""Matched C-series train/eval entry point; original run.py remains unchanged."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from string import Formatter
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

# Existing entry point sets up the pinned local CLIP and DINO dependencies.
from run import ManifestSplit, PROMPT
from combination_model import CSeriesSegmenter
from combination_core import Config, REGION_ALIGNMENT_DIRECTION_WEIGHT, VARIANTS, segmentation_loss
from isaid_protocol import development_epochs


class StatusSplit(ManifestSplit):
    def __getitem__(self, index):
        image, target, name = super().__getitem__(index)
        row = self.rows[index]
        status_path = row.get('pixel_status_file_name')
        if status_path:
            status_file = self.root / status_path
            with Image.open(status_file) as im:
                status = torch.from_numpy(np.array(im, dtype=np.int64))
        else:
            status = (target != self.ignore_id).long()
        if status.shape != target.shape or not ((status == 0) | (status == 1) | (status == 2)).all() or not torch.equal(status == 1, target != self.ignore_id):
            raise ValueError(f'Invalid annotation status for {name}')
        if (status==2).any():
            if not row.get('pixel_status_source') or row.get('pixel_status_sha256') != sha256(status_file.read_bytes()).hexdigest():
                raise ValueError(f'Verified negative pixels require recorded provenance and file hash: {name}')
        return image, target, status, name


def data_order(data, seed, epoch):
    return torch.randperm(len(data), generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def loader(data, args, order=None):
    return DataLoader(data, batch_size=args.batch_size, sampler=order, num_workers=args.workers,
                      pin_memory=torch.device(args.device).type == 'cuda', generator=torch.Generator().manual_seed(args.seed))


def training_policy(args, data):
    total = math.ceil(len(data) / args.batch_size) * args.epochs
    return {'epochs': args.epochs, 'planned_steps': total,
            'schedule': args.lr_schedule,
            'warmup_steps': min(args.train_warmup_steps, total // 10),
            'min_lr_ratio': args.min_lr_ratio, 'grad_clip': args.grad_clip}


def step_lr(base_lr, step, policy):
    """Optimizer-update indexed schedule; max_steps only interrupts, never changes horizon."""
    total, warm = policy['planned_steps'], policy['warmup_steps']
    if step < warm:
        return base_lr * (step + 1) / warm
    if policy['schedule'] == 'constant':
        return base_lr
    p = min(1., max(0., (step - warm) / max(1, total - warm - 1)))
    floor = policy['min_lr_ratio']
    return base_lr * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * p)))


def save(path, model, opt, meta, epoch, cursor, steps, seen, progress):
    device = next(model.parameters()).device
    state = {'metadata': meta, 'core': model.core.state_dict(), 'optimizer': opt.state_dict(),
             'epoch': epoch, 'cursor': cursor, 'steps': steps, 'seen': seen, 'progress': progress,
             'rng_cpu': torch.get_rng_state(),
             'rng_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
    atomic_save(path, state)


def atomic_save(path, state):
    # Never remove the previous recoverable file until the replacement is complete.
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def save_eval(path, model, meta, epoch, steps, seen):
    # Include all core buffers (including C5/C6 prototype memory), but no frozen
    # backbones, optimizer moments, RNG or training progress.
    atomic_save(path, {'metadata': meta, 'core': model.core.state_dict(),
                       'epoch': epoch, 'steps': steps, 'seen': seen,
                       'checkpoint_kind': 'evaluation_only'})


def blank_stats():
    return dict(updates=0, skipped_batches=0, loss=0., segmentation=0.,
                alignment=0., pseudo=0., auxiliary=0., grad_norm=0., clipped=0,
                pseudo_count=0, unknown_count=0, neg_pixels=0, kept=0, regions=0)


def stats_summary(s):
    n = max(1, s['updates'])
    return {**s, 'mean_loss': s['loss']/n, 'mean_segmentation_loss': s['segmentation']/n,
            'mean_weighted_alignment_loss': s['alignment']/n,
            'mean_weighted_pseudo_loss': s['pseudo']/n,
            'mean_weighted_auxiliary_loss': s['auxiliary']/n,
            'mean_preclip_grad_norm': s['grad_norm']/n,
            'clipped_fraction': s['clipped']/n, 'mean_train_kept': s['kept']/n}


def train(args, model, data, meta, state):
    model.train()
    text = model.text_features(data.classes)
    opt = torch.optim.AdamW(model.core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    policy = training_policy(args, data)
    if meta.get('training_policy') != policy:
        raise ValueError('Training policy metadata mismatch')
    start_epoch = start_cursor = steps = seen = 0
    progress = {'total': blank_stats(), 'current': blank_stats(), 'epochs': [], 'audit': {}}
    if state:
        if state.get('checkpoint_kind') == 'evaluation_only':
            raise ValueError('Evaluation-only checkpoint: exact training resume requires state.pt')
        for key in ('weights', 'size', 'prompt', 'manifest', 'data_sha256', 'train_classes', 'config', 'seed', 'batch_size', 'lr', 'weight_decay', 'ignore_id', 'limit', 'alignment_weight', 'region_to_text_loss_fraction', 'aux_weight', 'pseudo_weight', 'pseudo_warmup', 'schema', 'training_policy'):
            if state['metadata'].get(key) != meta.get(key):
                raise ValueError(f'Resume mismatch: {key}')
        if 'progress' not in state:
            raise ValueError('Old training checkpoint: evaluate it or resume with its pinned original commit')
        model.core.load_state_dict(state['core'], strict=True)
        opt.load_state_dict(state['optimizer'])
        start_epoch, start_cursor, steps, seen = (state[k] for k in ('epoch', 'cursor', 'steps', 'seen'))
        progress = state['progress']
        torch.set_rng_state(state['rng_cpu'])
        device = torch.device(args.device)
        cuda_rng = state['rng_cuda']
        if device.type == 'cuda':
            if isinstance(cuda_rng, list):  # Schema 2 states before device-local RNG capture.
                index = torch.cuda.current_device() if device.index is None else device.index
                cuda_rng = cuda_rng[index] if index < len(cuda_rng) else None
            if cuda_rng is None:
                raise ValueError('Training resume requires the checkpoint CUDA RNG state')
            torch.cuda.set_rng_state(cuda_rng, device)
    if args.max_steps and steps >= args.max_steps and start_epoch < args.epochs:
        raise ValueError('Checkpoint already at requested max_steps')
    t = perf_counter()
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    audit = progress['audit']
    frozen = [(p, p._version) for name, p in model.named_parameters() if not name.startswith('core.')]
    dev_epochs = development_epochs(args.epochs)
    for epoch in range(start_epoch, args.epochs):
        order = data_order(data, args.seed, epoch)
        cursor = start_cursor if epoch == start_epoch else 0
        for images, labels, status, ids in loader(data, args, order[cursor:]):
            cursor += len(images)
            if not (labels != data.ignore_id).any() and not (model.config.use_pixel and (status == 2).any()):
                for s in (progress['total'], progress['current']):
                    s['skipped_batches'] += 1
                continue
            images, labels, status = (x.to(args.device, non_blocking=True) for x in (images, labels, status))
            if args.audit and not audit:
                audit.update(model.audit_features(images[:1]))
            lr = step_lr(args.lr, steps, policy)
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            r = model(images, text, labels=labels, ignore=data.ignore_id, status=status,
                      pseudo_enabled=steps >= args.pseudo_warmup, alignment_enabled=args.alignment_weight > 0)
            seg = segmentation_loss(r['logits'], labels, data.ignore_id, model.config.use_pixel, status)
            zero = seg.new_zeros(())
            align = args.alignment_weight * r.get('alignment_loss', zero)
            pseudo = args.pseudo_weight * r.get('pseudo_loss', zero)
            aux = args.aux_weight * segmentation_loss(r['aux_logits'], labels, data.ignore_id) if 'aux_logits' in r else zero
            loss = seg + align + pseudo + aux
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss on {ids}')
            loss.backward()
            if args.audit and steps < 3:
                audit[f'grad_step_{steps + 1}'] = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in model.core.named_parameters()}
            # Also checks nonfinite gradients when clipping is disabled.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.core.parameters(),
                             args.grad_clip if args.grad_clip > 0 else float('inf'), error_if_nonfinite=True))
            opt.step()
            if model.config.use_rs:
                model.core.memory.update(r['regions'], r['region_ids'])
            steps += 1; seen += len(images)
            values = torch.stack((loss.detach(), seg.detach(), align.detach(), pseudo.detach(), aux.detach())).cpu().tolist()
            update = dict(zip(('loss', 'segmentation', 'alignment', 'pseudo', 'auxiliary'), values))
            update.update(updates=1, grad_norm=grad_norm, clipped=int(args.grad_clip > 0 and grad_norm > args.grad_clip),
                          pseudo_count=int(r.get('pseudo_count',0)), unknown_count=int(r.get('unknown_count',0)),
                          neg_pixels=int((status==2).sum()), kept=len(r['kept']), regions=len(r.get('region_ids', [])))
            for s in (progress['total'], progress['current']):
                for key, value in update.items():
                    s[key] += value
            if args.audit and steps == 1:
                audit['first_train_ids'] = list(ids)
            if steps == 1 or steps % 100 == 0:
                print(f'step={steps} epoch={epoch+1} lr={lr:.8g} loss={values[0]:.6f} seg={values[1]:.6f} align={values[2]:.6f} pseudo={values[3]:.6f} aux={values[4]:.6f} preclip_norm={grad_norm:.6f}', flush=True)
            if steps % args.save_every == 0 or (args.max_steps and steps >= args.max_steps):
                save(args.output / 'state.pt', model, opt, meta, epoch, cursor, steps, seen, progress)
            if args.max_steps and steps >= args.max_steps:
                break
        if cursor == len(order):
            summary = {'epoch': epoch+1, 'steps': steps, 'seen_images': seen,
                       'last_lr': opt.param_groups[0]['lr'], **stats_summary(progress['current'])}
            progress['epochs'].append(summary)
            progress['current'] = blank_stats()
            # Keep lightweight exports for the scheduled dev curve and a
            # separate final export for full validation.
            if epoch+1 == args.epochs:
                save_eval(args.output / 'final.pt', model, meta, epoch+1, steps, seen)
            elif epoch+1 in dev_epochs:
                save_eval(args.output / f'epoch_{epoch+1:03d}.pt', model, meta, epoch+1, steps, seen)
            # Write snapshots first: a crash cannot advance the recovery cursor
            # past an epoch whose required evaluation export does not exist.
            save(args.output / 'state.pt', model, opt, meta, epoch+1, 0, steps, seen, progress)
            print('epoch_summary=' + json.dumps(summary, allow_nan=False), flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    if not steps:
        raise RuntimeError('No optimizer steps executed')
    assert all(p.grad is None and p._version == version for p, version in frozen), 'Frozen encoder mutated'
    s = stats_summary(progress['total'])
    return {**meta, 'steps': steps, 'seen_images': seen, 'completed_epochs': len(progress['epochs']),
            'epoch_history': progress['epochs'], 'train_statistics': s, 'mean_loss': s['mean_loss'],
            'wall_seconds_this_invocation': perf_counter() - t, 'audit': audit,
            'train_peak_cuda_mib_this_invocation': torch.cuda.max_memory_allocated(device)/2**20 if cuda else None,
            'train_peak_reserved_cuda_mib_this_invocation': torch.cuda.max_memory_reserved(device)/2**20 if cuda else None,
            'pseudo_selected_patches': s['pseudo_count'], 'wholly_unknown_patches': s['unknown_count'],
            'verified_negative_pixels': s['neg_pixels'], 'mean_train_kept': s['mean_train_kept'], 'region_count': s['regions']}


def mean_for(iou, indices):
    vals = [iou[i] for i in indices if iou[i] is not None]
    return sum(vals) / len(vals) if vals else None


def inference_passes(args, model, images, text):
    sizes = getattr(args, 'eval_sizes', None)
    rotations = getattr(args, 'eval_rotations', (0,))
    native_weight = getattr(args, 'native_clip_weight', 0.)
    if sizes is None and rotations == (0,) and not native_weight:
        return model(images, text, force_keep_all=args.force_keep_all)

    original_size = model.size
    scores = None
    sizes = sizes or (original_size,)
    try:
        for size in sizes:
            model.size = size
            for angle in rotations:
                turns = angle // 90
                rotated = torch.rot90(images, turns, (-2, -1)) if turns else images
                if native_weight < 1.:
                    result = model(rotated, text, force_keep_all=args.force_keep_all)
                    logits, kept = result['logits'], result['kept']
                if native_weight:
                    patches = model.clip_tokens(model.resize_rgb(rotated))[:, 1:]
                    native = model.pixel_logits(patches, text, rotated.shape[-2:])
                    logits = torch.lerp(logits, native, native_weight) if native_weight < 1. else native
                    kept = torch.arange(len(text), device=native.device)
                aligned = torch.rot90(logits, -turns, (-2, -1)) if turns else logits
                scores = aligned.clone() if scores is None else scores.add_(aligned)
    finally:
        model.size = original_size
    return {'logits': scores.div_(len(sizes) * len(rotations)), 'kept': kept}


@torch.no_grad()
def evaluate(args, model, data, meta):
    model.eval()
    prompts = getattr(args, 'eval_prompts', None)
    text = model.text_features(data.classes, prompts) if prompts else model.text_features(data.classes)
    k = len(data.classes)
    confusion = torch.zeros(k, k, dtype=torch.long, device=args.device)
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    if cuda:
        warm = torch.stack([data[i][0] for i in range(min(len(data), args.batch_size))]).to(args.device)
        for _ in range(args.warmup):
            inference_passes(args, model, warm, text)['logits'].argmax(1)
        torch.cuda.synchronize(device); del warm
        torch.cuda.reset_peak_memory_stats(device)
    times, sizes = [], []; present_classes = retained_classes = retained_pixels = total_pixels = 0
    compute_loss = getattr(args, 'eval_loss', False)
    loss_sum = torch.zeros((), dtype=torch.float64, device=args.device) if compute_loss else None
    seen = set(model.train_classes); sample_digest = sha256(); peak = 0
    sample_order = None
    if args.eval_samples is not None:
        sample_order = sorted(data_order(data, args.eval_sample_seed, 0)[:args.eval_samples])
    for images, labels, _, names in loader(data, args, sample_order):
        images = images.to(args.device, non_blocking=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
        else:
            start = perf_counter()
        r = inference_passes(args, model, images, text)
        pred = r['logits'].argmax(1)
        if cuda:
            end.record(stream)
            end.synchronize()
            times.append(start.elapsed_time(end))
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        else:
            times.append((perf_counter() - start) * 1000)
        if not torch.isfinite(r['logits']).all():
            raise RuntimeError(f'Nonfinite evaluation logits on {names}')
        keep = r['kept']; sizes.extend([len(keep)] * len(images))
        labels = labels.to(args.device); valid = labels != data.ignore_id
        if compute_loss:
            loss_sum += F.cross_entropy(r['logits'], labels, ignore_index=data.ignore_id, reduction='sum')
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
    report = {**meta, 'test_classes': data.classes, 'miou_percent': mean_for(iou, range(k)),
            'seen_miou': si, 'unseen_miou': ui, 'harmonic_iou': harmonic,
            'class_iou': dict(zip(data.classes, iou)), 'confusion_matrix': confusion.tolist(),
            'sample_ids_sha256': sample_digest.hexdigest(), 'images': len(sizes),
            'mean_retained_classes': sum(sizes)/len(sizes), 'class_presence_recall': retained_classes/present_classes if present_classes else None,
            'pixel_recall': retained_pixels/total_pixels if total_pixels else None,
            'batch_ms_mean': sum(times)/len(times), 'batch_ms_p95': float(np.percentile(times,95)),
            'ms_per_image': sum(times)/len(sizes), 'peak_cuda_mib': peak/2**20 if cuda else None,
            'force_keep_all': args.force_keep_all, 'eval_samples': args.eval_samples,
            'eval_sample_seed': args.eval_sample_seed if args.eval_samples is not None else None}
    if compute_loss:
        report['mean_pixel_cross_entropy'] = loss_sum.item() / total_pixels if total_pixels else None
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train','eval'))
    p.add_argument('--variant', choices=VARIANTS)
    for name in ('manifest','clip-weights','dino-weights','output','checkpoint'):
        p.add_argument('--'+name, type=Path, required=name in ('manifest','clip-weights','output'))
    p.add_argument('--device', default='cuda'); p.add_argument('--size',type=int,help='Training encoder size (384 by default); evaluation defaults to the recorded size')
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--workers',type=int,default=0)
    p.add_argument('--limit',type=int); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=5); p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--lr-schedule',choices=('constant','cosine'),default='cosine')
    p.add_argument('--train-warmup-steps',type=int,default=300)
    p.add_argument('--min-lr-ratio',type=float,default=.1); p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--eval-samples',type=int); p.add_argument('--eval-sample-seed',type=int,default=42)
    p.add_argument('--data-sha256', help='Content identity from the audited source release')
    p.add_argument('--eval-loss', action='store_true', help='Report pixel cross-entropy over the evaluation vocabulary')
    p.add_argument('--eval-sizes', type=int, nargs='+', help='Evaluate at these sizes, averaging native-grid pixel logits across passes')
    p.add_argument('--eval-prompt', dest='eval_prompts', action='append', help='Template containing one {}; repeat to average normalized text embeddings')
    p.add_argument('--native-clip-weight', type=float, default=0., help='Weight of native CLIP patch logits in an inference-only blend')
    p.add_argument('--eval-rotations', type=int, choices=(0, 90, 180, 270), nargs='+', help='Rotate inputs, invert output rotations and average logits')
    p.add_argument('--weight-decay',type=float,default=.01); p.add_argument('--max-steps',type=int)
    p.add_argument('--save-every',type=int,default=500); p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--audit',action='store_true'); p.add_argument('--force-keep-all',action='store_true')
    p.add_argument('--affinity',choices=('visual','semantic','dual'),default='dual')
    p.add_argument('--fusion',choices=('context','separate','learned','mean'),default='context')
    p.add_argument('--region-mode',choices=('component','class'),default='component')
    p.add_argument('--relation-source',choices=('native','fused','joint'),default='joint')
    p.add_argument('--relation-control',choices=('global','conditional'),default='conditional')
    p.add_argument('--relation-chunk',type=int,default=128)
    p.add_argument('--router',choices=('union','dual'),default=None)
    p.add_argument('--route-train',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--pseudo',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--pseudo-conf',type=float,default=.9)
    p.add_argument('--pseudo-margin',type=float,default=.2)
    p.add_argument('--pseudo-weight',type=float,default=.05)
    p.add_argument('--pseudo-warmup',type=int,default=200)
    p.add_argument('--decoder',choices=('guided','shared'),default='guided')
    p.add_argument('--neighbors',type=int,default=32); p.add_argument('--topk',type=int)
    p.add_argument('--rs',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--pixel',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--alignment-weight',type=float,default=.2); p.add_argument('--aux-weight',type=float,default=.2)
    args=p.parse_args()
    if args.data_sha256 is not None and (len(args.data_sha256) != 64 or
                                         any(c not in '0123456789abcdef' for c in args.data_sha256)):
        p.error('--data-sha256 requires a lowercase SHA-256 digest')
    if args.batch_size<1 or args.save_every<1 or args.epochs<1 or args.alignment_weight<0 or args.aux_weight<0 or args.pseudo_weight<0 or args.pseudo_warmup<0: p.error('Invalid training settings')
    if args.lr <= 0 or args.train_warmup_steps < 0 or not 0 <= args.min_lr_ratio <= 1 or args.grad_clip < 0: p.error('Invalid optimizer policy')
    if args.eval_samples is not None and (args.mode!='eval' or args.eval_samples<1 or args.limit is not None): p.error('Use positive --eval-samples only in eval, without --limit')
    if args.mode != 'eval' and (args.eval_loss or args.eval_sizes or args.eval_rotations
                               or args.eval_prompts or args.native_clip_weight):
        p.error('Evaluation diagnostics are evaluation only')
    args.eval_sizes = tuple(args.eval_sizes) if args.eval_sizes else None
    args.eval_rotations = tuple(args.eval_rotations or (0,))
    if args.eval_sizes and (any(size <= 0 or size % 16 for size in args.eval_sizes)
                            or len(set(args.eval_sizes)) != len(args.eval_sizes)):
        p.error('--eval-sizes requires distinct positive multiples of 16')
    if len(set(args.eval_rotations)) != len(args.eval_rotations):
        p.error('--eval-rotations requires distinct angles')
    if not 0. <= args.native_clip_weight <= 1.:
        p.error('--native-clip-weight must lie between 0 and 1')
    for prompt in args.eval_prompts or ():
        try:
            fields = [(name, spec, conv) for _, name, spec, conv in Formatter().parse(prompt) if name is not None]
        except ValueError:
            fields = []
        if fields != [('', '', None)]:
            p.error('--eval-prompt requires one unformatted {} class placeholder')
    if args.eval_prompts and len(set(args.eval_prompts)) != len(args.eval_prompts):
        p.error('--eval-prompt templates must be distinct')
    if args.mode=='eval' and not args.checkpoint: p.error('Evaluation requires checkpoint')
    if args.mode=='train' and args.force_keep_all: p.error('--force-keep-all is evaluation only; use --no-route-train for full-vocabulary training')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    state = checkpoint_sha256 = None
    if args.checkpoint:
        with args.checkpoint.open('rb') as checkpoint:
            if args.mode == 'eval':
                digest = sha256()
                for chunk in iter(lambda: checkpoint.read(4 << 20), b''):
                    digest.update(chunk)
                checkpoint_sha256 = digest.hexdigest()
                checkpoint.seek(0)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state is not None and state['metadata'].get('schema')!=2: p.error('V1 checkpoint: use pinned V1 commit; V2 parameters differ')
    if args.mode=='train' and state is not None and state.get('checkpoint_kind')=='evaluation_only':
        p.error('Evaluation-only checkpoint cannot resume optimizer/RNG; use state.pt')
    if args.size is None:
        args.size = state['metadata']['size'] if args.mode == 'eval' else 384
    if args.mode=='eval':
        cfg=Config(**state['metadata']['config']); train_classes=state['metadata']['train_classes']
        multipass = len(args.eval_sizes or (args.size,)) * len(args.eval_rotations) > 1
        if cfg.variant == 'C6' and (multipass or args.native_clip_weight) and not args.force_keep_all:
            p.error('C6 multi-pass or native-CLIP logit averaging requires --force-keep-all')
        if args.variant and args.variant!=cfg.variant: p.error('Variant differs from checkpoint')
        if args.topk is not None or args.router is not None:
            if cfg.variant!='C6': p.error('Routing overrides require C6')
            cfg=Config(**{**asdict(cfg),'topk':args.topk if args.topk is not None else cfg.topk,'router':args.router or cfg.router})
    else:
        if not args.variant: p.error('Training requires variant')
        cfg=Config(variant=args.variant,affinity=args.affinity,neighbors=args.neighbors,topk=args.topk if args.topk is not None else 4,rs=args.rs,pixel=args.pixel,fusion=args.fusion,region_mode=args.region_mode,relation_source=args.relation_source,relation_control=args.relation_control,router=args.router or 'dual',route_train=args.route_train,pseudo=args.pseudo,pseudo_conf=args.pseudo_conf,pseudo_margin=args.pseudo_margin,relation_chunk=args.relation_chunk,decoder=args.decoder)
        train_classes=None
    split='train' if args.mode=='train' else 'val'; data=StatusSplit(args.manifest,split,args.limit)
    if not len(data) or len(set(data.classes))!=len(data.classes): p.error('Empty split or duplicate classes')
    if train_classes is None: train_classes=data.classes
    model=CSeriesSegmenter(args.clip_weights.expanduser(),args.dino_weights.expanduser() if args.dino_weights else None,args.size,cfg,train_classes).to(args.device)
    meta={'schema':2,'config':asdict(cfg),'weights':model.weight_hashes,'size':args.size,'prompt':PROMPT,
          'manifest':sha256(args.manifest.read_bytes()).hexdigest(),'train_classes':train_classes,
          'ignore_id':data.ignore_id,'seed':args.seed,'batch_size':args.batch_size,'lr':args.lr,
          'weight_decay':args.weight_decay,'alignment_weight':args.alignment_weight,'aux_weight':args.aux_weight,'pseudo_weight':args.pseudo_weight,'pseudo_warmup':args.pseudo_warmup,
          'limit':args.limit,'split':split,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'torch':str(torch.__version__),'device':str(args.device),'dtype':'float32','tf32':False, 'checkpoint_train_manifest': state['metadata']['manifest'] if state else None}
    if args.data_sha256 is not None:
        meta['data_sha256'] = args.data_sha256
    meta['training_policy'] = training_policy(args, data) if args.mode=='train' else state['metadata'].get('training_policy')
    if args.mode == 'eval':
        for key in ('alignment_weight', 'aux_weight', 'pseudo_weight', 'pseudo_warmup'):
            meta[key] = state['metadata'].get(key, meta[key])
    if cfg.variant == 'F08':
        # Older F08 exports can still be evaluated with their original loss provenance.
        meta['region_to_text_loss_fraction'] = (REGION_ALIGNMENT_DIRECTION_WEIGHT if args.mode == 'train'
                                               else state['metadata'].get('region_to_text_loss_fraction', 1.))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='train':
        if (args.output/'state.pt').exists() and not args.checkpoint: p.error('Refusing to overwrite a prior run; use a new directory or resume')
        report=train(args,model,data,meta,state)
    else:
        for key in ('weights','size','prompt'):
            if state['metadata'][key]!=meta[key]: raise ValueError(f'Eval mismatch: {key}')
        model.core.load_state_dict(state['core'],strict=True); report=evaluate(args,model,data,meta)
        report['checkpoint_epoch'] = state.get('epoch')
        report['checkpoint_steps'] = state.get('steps')
        report['checkpoint_kind'] = state.get('checkpoint_kind', 'training_state')
        report['checkpoint_sha256'] = checkpoint_sha256
        report['checkpoint_training'] = {
            key: state['metadata'][key] for key in (
                'manifest', 'data_sha256', 'weights', 'config', 'seed', 'batch_size', 'lr', 'weight_decay', 'size',
                'limit', 'training_policy', 'alignment_weight', 'region_to_text_loss_fraction',
                'aux_weight', 'pseudo_weight', 'pseudo_warmup',
            ) if key in state['metadata']
        }
        report['inference_sizes'] = args.eval_sizes or (args.size,)
        report['inference_rotations'] = args.eval_rotations
        report['inference_prompts'] = args.eval_prompts or (PROMPT,)
        report['native_clip_weight'] = args.native_clip_weight
    # A partial JSON file must never be mistaken for a completed job/cleanup signal.
    report_tmp = args.output/'report.tmp'
    report_tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report_tmp.replace(args.output/'report.json')
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)

if __name__=='__main__': main()
