"""Run from repository root: python -m research.ov_routes.run --help."""
import argparse
import json
import math
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F
from .data import Manifest, augment, digest, episode, metrics
from .episodic import objective, masked_bce, remap_support
from .host import build_host
from .views import ViewSelector, predict_views, training_actions


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore(model, saved):
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(saved) != expected:
        raise ValueError('Trainable state keys differ')
    with torch.no_grad():
        for n, value in saved.items():
            if value.shape != params[n].shape:
                raise ValueError('Checkpoint shape mismatch: '+n)
            params[n].copy_(value)


@torch.no_grad()
def evaluate(host, data, device, policy='fixed', budget=0, selector=None, seed=42):
    cm = torch.zeros(len(data.all_names), len(data.all_names), dtype=torch.long)
    records = []
    rng = torch.Generator().manual_seed(seed)
    for i in range(len(data.rows['val'])):
        image, labels = data.read('val', i)
        scores, trace = predict_views(host, image.to(device), data.all_names,
                                      policy, budget, selector, rng)
        pred = scores.argmax(0).cpu()
        valid = labels != 255
        one = torch.bincount(labels[valid]*len(cm)+pred[valid], minlength=cm.numel()).reshape_as(cm)
        cm += one
        records.append({'index': i, 'file': data.rows['val'][i]['file_name'],
                        **trace, **metrics(one, data.seen)})
    return {**metrics(cm, data.seen), 'class_names': data.all_names,
            'policy': policy, 'budget': budget, 'per_image': records}


def check(host, data, device):
    rng = torch.Generator().manual_seed(901)
    si, qi, held = episode(data, rng)
    x, y = augment(*data.read('train', si), rng)
    q, z = augment(*data.read('train', qi), rng)
    x, y, q, z = [t.to(device) for t in (x, y, q, z)]
    parity = host.check_native(x, data.names)
    # Full host gradients and class-subset fast weights must work before long training.
    before = state(host)
    loss, detail = objective(host, x[None], y[None], q[None], z[None], data.names, held)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError('Real-image meta preflight failed')
    loss.backward()
    norm = sum(float(p.grad.abs().sum()) for p in host.parameters() if p.grad is not None)
    if norm == 0 or not math.isfinite(norm) or any(p.grad is not None for p in host.clip.parameters()):
        raise RuntimeError('Invalid full-host/frozen-CLIP gradients')
    # Isolate the query contribution to the outer gradient and verify the
    # temporary support update itself is nonzero on actual pretrained weights.
    params = [p for p in host.parameters() if p.requires_grad]
    outer = [None if p.grad is None else p.grad.detach().clone() for p in params]
    support_ids = [i for i in range(len(data.names)) if i not in held]
    support = masked_bce(host(x[None], [data.names[i] for i in support_ids]),
                         remap_support(y[None], support_ids))
    support_grad = torch.autograd.grad(support, params, allow_unused=True)
    inner_norm = sum(float(g.abs().sum()) for g in support_grad if g is not None)
    query_norm = 0.
    for outer_grad, inner_grad, param in zip(outer, support_grad, params):
        if outer_grad is not None or inner_grad is not None:
            u = torch.zeros_like(param) if outer_grad is None else outer_grad
            v = torch.zeros_like(param) if inner_grad is None else inner_grad
            query_norm += float((u-v).abs().sum())
    if not math.isfinite(inner_norm) or not math.isfinite(query_norm) or inner_norm == 0 or query_norm == 0:
        raise RuntimeError('Real support update or held-class query gradient is zero')
    for n, p in host.named_parameters():
        if n in before:
            torch.testing.assert_close(p.detach().cpu(), before[n], rtol=0, atol=0)
    host.zero_grad(set_to_none=True)
    sample = training_actions(host, q, z, data.names)
    if sample is None or not torch.isfinite(sample[1]).all():
        raise RuntimeError('Real-image crop enumeration failed')
    no_views, trace0 = predict_views(host, q, data.names, budget=0)
    torch.testing.assert_close(no_views, host.predict_logits(q[None], data.names)[0].sigmoid())
    _, trace1 = predict_views(host, q, data.names, budget=1)
    return {'native_max_error': parity, 'meta_loss': float(loss.detach()),
            'gradient_l1': norm, 'support_gradient_l1': inner_norm,
            'query_gradient_l1': query_norm, 'episode': [si, qi], **detail,
            'view_gains': sample[1].tolist(), 'zero_views': trace0, 'one_view': trace1}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['preflight', 'train-a', 'train-b', 'evaluate'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True, help='Matching full CAT-Seg ViT-B/16 YAML')
    p.add_argument('--weights', required=True, help='Complete pretrained checkpoint, never a 267k prototype')
    p.add_argument('--legacy-weights', action='store_true', help='Permit absent config metadata; tensor loading remains strict')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, help='Required total updates for training; no hidden short pilot default')
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--inner-lr', type=float, default=1e-3)
    p.add_argument('--weight', type=float, default=1.)
    p.add_argument('--mode', choices=['meta', 'joint', 'erm'], default='meta')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--policy', choices=['fixed', 'random', 'learned'], default='fixed')
    p.add_argument('--budget', type=int, default=1)
    p.add_argument('--artifact', help='Route A adaptation or route B selector for evaluation')
    p.add_argument('--resume', help='Restore a same-protocol training checkpoint into a NEW output directory')
    return p


def main():
    p = parser()
    a = p.parse_args()
    training = a.task.startswith('train-')
    if training and (a.steps is None or a.steps < 1):
        p.error('Training needs explicit positive --steps')
    if a.lr <= 0 or a.inner_lr <= 0 or a.weight <= 0 or a.eval_every < 1 or not 0 <= a.budget <= 8:
        p.error('Invalid numeric setting')
    if a.artifact and a.task != 'evaluate':
        p.error('--artifact is evaluation-only; use --resume for continuation')
    if a.resume and not training:
        p.error('--resume is training-only')
    if a.task == 'evaluate' and a.policy == 'learned' and not a.artifact:
        p.error('Learned policy needs a trained selector artifact')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    data = Manifest(a.manifest)
    host, cfg = build_host(a.config, a.weights, data.train_classes, data.test_classes,
                           a.device, a.legacy_weights)
    source = {f.name: digest(f) for f in Path(__file__).parent.glob('*.py')}
    repository = Path(__file__).resolve().parents[2]
    core_source = {str(f.relative_to(repository)): digest(f)
                   for f in (repository/'cat_seg').rglob('*.py')}
    identity = {**data.identity, 'weights': digest(a.weights), 'resolved_config': cfg,
                'source': source, 'host_source': core_source}
    protocol = {'version': 1, 'identity': identity, 'task': a.task, 'mode': a.mode,
                'seed': a.seed, 'steps': a.steps, 'lr': a.lr, 'inner_lr': a.inner_lr,
                'weight': a.weight, 'eval_every': a.eval_every, 'budget': a.budget,
                'names': data.names, 'seen_ids': data.seen,
                'precision': 'fp32', 'frozen_clip': True, 'module_mode': 'eval',
                'cudnn_tf32': False, 'matmul_tf32': False,
                'deterministic_algorithms': True, 'cublas_workspace': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                'train_images_per_a_update': 2, 'gain_scale': 100.}
    write(out/'protocol.json', protocol)
    if a.artifact:
        c = torch.load(a.artifact, map_location='cpu', weights_only=True)
        if c['protocol']['identity'] != identity:
            raise ValueError('Artifact host/config/data/code identity mismatch')
        if c['protocol']['task'] == 'train-a':
            if a.policy == 'learned':
                raise ValueError('Route A artifact is not a selector')
            restore(host, c['parameters'])
            selector = None
        elif c['protocol']['task'] == 'train-b':
            selector = ViewSelector(host.text_vectors(data.names).shape[-1]).to(a.device)
            restore(selector, c['parameters'])
        else:
            raise ValueError('Unknown artifact task')
    else:
        selector = None
    if a.task == 'evaluate':
        result = evaluate(host, data, a.device, a.policy, a.budget, selector, a.seed)
        result.update(artifact=a.artifact, artifact_sha256=digest(a.artifact) if a.artifact else None)
        write(out/'validation.json', result)
        print(json.dumps({k: result[k] for k in ('miou', 'seen', 'unseen', 'hiou')}))
        return
    # Always run REAL input/weight preflight before training. Does not update weights.
    write(out/'preflight.json', check(host, data, a.device))
    if a.task == 'preflight':
        return
    torch.manual_seed(a.seed)  # Preflight cannot perturb training init/sampling.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    rng = torch.Generator().manual_seed(a.seed+1)
    aug_rng = torch.Generator().manual_seed(a.seed+2)
    if a.task == 'train-b':
        host.requires_grad_(False)
        selector = ViewSelector(host.text_vectors(data.names).shape[-1]).to(a.device)
        optimized = selector
    else:
        optimized = host
    optimizer = torch.optim.AdamW([x for x in optimized.parameters() if x.requires_grad],
                                  lr=a.lr, weight_decay=.01)
    start = 0
    if a.resume:
        c = torch.load(a.resume, map_location='cpu', weights_only=True)
        if c['protocol'] != protocol:
            raise ValueError('Resume protocol mismatch; total --steps must be unchanged')
        restore(optimized, c['parameters'])
        optimizer.load_state_dict(c['optimizer'])
        rng.set_state(c['rng']); aug_rng.set_state(c['aug_rng'])
        torch.set_rng_state(c['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(c['cuda_rng'])
        start = c['updates']
        if start >= a.steps:
            raise ValueError('Checkpoint has already reached the requested endpoint')
    else:
        write(out/'validation_000000.json', evaluate(host, data, a.device, budget=0))
    eligible = [i for i, ids in enumerate(data.support()) if ids]
    if not eligible:
        raise ValueError('No valid source training pixels')
    with (out/'train.jsonl').open('w') as log:
        update, attempts = start, 0
        while update < a.steps:
            attempts += 1
            if attempts > 20*a.steps:
                raise RuntimeError('Too many empty episodes; inspect source labels and resize')
            optimizer.zero_grad(set_to_none=True)
            if a.task == 'train-a':
                s, q, held = episode(data, rng)
                x, y = augment(*data.read('train', s), aug_rng)
                v, z = augment(*data.read('train', q), aug_rng)
                x, y, v, z = [t.to(a.device) for t in (x, y, v, z)]
                loss, detail = objective(host, x[None], y[None], v[None], z[None], data.names,
                                         held, a.inner_lr, a.weight, a.mode)
                detail['images'] = [s, q]
            else:
                i = eligible[int(torch.randint(len(eligible), (), generator=rng))]
                image, labels = data.read('train', i)
                sample = training_actions(host, image.to(a.device), labels.to(a.device), data.names)
                if sample is None:
                    loss, detail = None, {'skip': 'no_valid_pixels'}
                else:
                    features, gain = sample
                    loss = F.mse_loss(selector(features), gain*100.)
                    detail = {'images': [i], 'gain_min': float(gain.min()), 'gain_max': float(gain.max())}
            if loss is None:
                log.write(json.dumps({'attempt': attempts, **detail})+'\n')
                continue  # No AdamW decay/momentum update on empty data.
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimized.parameters(), 1., error_if_nonfinite=True)
            warmup = min(100, max(1, a.steps//20))
            step = update+1
            lr = a.lr*(step/warmup if step <= warmup else
                       .01+.99*.5*(1+math.cos(math.pi*(step-warmup)/max(1, a.steps-warmup))))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            update += 1
            record = {'update': update, 'loss': float(loss.detach()), 'lr': lr, **detail}
            log.write(json.dumps(record)+'\n'); log.flush()
            if update % 20 == 0:
                print(json.dumps(record), flush=True)
            if update % a.eval_every == 0 or update == a.steps:
                c = {'protocol': protocol, 'updates': update, 'parameters': state(optimized),
                     'optimizer': optimizer.state_dict(), 'rng': rng.get_state(),
                     'aug_rng': aug_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
                tmp = out/'checkpoint.tmp'
                torch.save(c, tmp)
                tmp.replace(out/f'checkpoint_{update:06d}.pt')
                result = evaluate(host, data, a.device, 'learned' if selector else 'fixed',
                                  a.budget if selector else 0, selector, a.seed)
                write(out/f'validation_{update:06d}.json', result)
    write(out/'COMPLETE.json', {'updates': update, 'task': a.task,
                               'note': 'No best-checkpoint selection; inspect fixed endpoint and trajectory.'})


if __name__ == '__main__':
    main()
