"""Matched fixed-update pilots on audited frozen-feature caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from data import load_cache, check_compatible, batch, sha256, write_json
from evaluate import evaluate
from model import SemanticStructureDecoder, optimization_step, MODEL_VERSION


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def run(a):
    if min(a.steps, a.batch_size, a.width, a.save_every, a.eval_every, a.threads) < 1 or a.lr <= 0:
        raise ValueError('Invalid training budget/configuration')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    if str(a.device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.init()
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats(a.device)
    train = load_cache(a.cache, 'train')
    val = load_cache(a.val_cache, 'validation') if a.val_cache else None
    if val is not None:
        check_compatible(train, val)
        if train['metadata']['dataset'] != val['metadata']['dataset'] or train['metadata']['class_names'] != val['metadata']['class_names']:
            raise ValueError('Training validation must be source-dataset validation with identical class IDs; use evaluate.py for transfer')
        if not torch.equal(train['text'], val['text']) or not torch.equal(train['valid'], val['valid']):
            raise ValueError('Source train/validation text bank mismatch')
    eligible = (train['labels'] != 255).flatten(1).any(1).nonzero().flatten()
    if not len(eligible):
        raise ValueError('No supervised training images')
    config = dict(clip_dim=train['visual'].shape[1], structure_dims=[s.shape[1] for s in train['structures']],
                  width=a.width, semantic_mode=a.semantic, spatial_mode=a.spatial,
                  temperature=a.temperature, semantic_strength=a.semantic_strength)
    model = SemanticStructureDecoder(**config).to(a.device)
    initial_sha = state_digest(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sampler = torch.Generator().manual_seed(a.seed + 1)
    cache_sha = sha256(a.cache)
    val_sha = sha256(a.val_cache) if a.val_cache else None
    source_hashes = {n: sha256(Path(__file__).with_name(n)) for n in ('model.py','data.py','evaluate.py','train_cached.py')}
    controls = dict(seed=a.seed, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
                    cache_sha256=cache_sha, val_cache_sha256=val_sha, source_sha256=source_hashes,
                    device=str(a.device), threads=a.threads)
    updates, trace, history = 0, [], []
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=True)
        if saved.get('model_version') != MODEL_VERSION or saved['config'] != config or saved['controls'] != controls:
            raise ValueError('Resume requires identical model, cache, code and training controls')
        model.load_state_dict(saved['model'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        sampler.set_state(saved['sampler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        if saved['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        updates, trace, history = saved['updates'], saved['sample_trace'], saved['history']
        initial_sha = saved['initial_state_sha256']
        if updates >= a.steps:
            raise ValueError('--steps must exceed resumed update count')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'model_version': MODEL_VERSION, 'config': config, 'controls': controls,
                'requested_optimizer_updates': a.steps, 'checkpoint_selection': 'fixed final only',
                'all_ignore_images_excluded': len(train['visual']) - len(eligible),
                'eligible_training_images': len(eligible), 'sampling': 'with replacement; separate CPU generator',
                'initial_state_sha256': initial_sha, 'parameters': sum(p.numel() for p in model.parameters()),
                'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                'cuda': torch.version.cuda, 'device': str(a.device)},
                'resume': str(a.resume) if a.resume else None}
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'train_metadata.json', train['metadata'])
    start = time.perf_counter()
    def save(path):
        atomic_save({'model_version': MODEL_VERSION, 'config': config, 'controls': controls,
                     'model': model.state_dict(), 'optimizer': opt.state_dict(), 'updates': updates,
                     'sampler_state': sampler.get_state(), 'torch_rng_state': torch.get_rng_state(),
                     'cuda_rng_state': torch.cuda.get_rng_state_all() if str(a.device).startswith('cuda') else None,
                     'sample_trace': trace, 'history': history, 'initial_state_sha256': initial_sha,
                     'train_signature': train['metadata']}, path)
    model.train()
    with (out / 'train.jsonl').open('w') as log:
        while updates < a.steps:
            indices = eligible[torch.randint(len(eligible), (a.batch_size,), generator=sampler)]
            if str(a.device).startswith('cuda'):
                torch.cuda.synchronize(a.device)
            step_start = time.perf_counter()
            loss = optimization_step(model, opt, *batch(train, indices, a.device))
            if str(a.device).startswith('cuda'):
                torch.cuda.synchronize(a.device)
            step_seconds = time.perf_counter() - step_start
            if loss is None:
                raise RuntimeError('Eligible-image sampler unexpectedly produced all-ignore batch')
            updates += 1
            trace.append(indices.tolist())
            record = {'update': updates, 'loss': loss, 'indices': indices.tolist(),
                      'step_seconds': step_seconds}
            history.append(record)
            log.write(json.dumps(record) + '\n'); log.flush()
            if updates == 1 or updates % 10 == 0:
                print(json.dumps(record), flush=True)
            if val is not None and (updates % a.eval_every == 0 or updates == a.steps):
                result = evaluate(model, val, a.device, a.batch_size)
                result['update'] = updates
                write_json(out / f'validation_{updates:06d}.json', result)
            if updates % a.save_every == 0:
                save(out / 'latest.pt')
    save(out / 'final.pt')
    # Verify persistence via an independent module instance and deterministic eval inputs.
    restored = SemanticStructureDecoder(**config).to(a.device)
    reloaded = torch.load(out / 'final.pt', map_location=a.device, weights_only=True)
    restored.load_state_dict(reloaded['model'], strict=True)
    model.eval(); restored.eval()
    with torch.no_grad():
        v, t, mask, structure, labels = batch(train, eligible[:1], a.device)
        expected = model(v, t, mask, structure, labels.shape[-2:])
        actual = restored(v, t, mask, structure, labels.shape[-2:])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if str(a.device).startswith('cuda'):
        torch.cuda.synchronize(a.device)
    write_json(out / 'summary.json', {'updates': updates, 'initial_state_sha256': initial_sha,
               'final_state_sha256': state_digest(model.state_dict()),
               'sample_trace_sha256': hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
               'reload_exact': True, 'elapsed_seconds_this_invocation': time.perf_counter()-start,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device) if str(a.device).startswith('cuda') else None,
               'validation_used': val is not None, 'claim': 'engineering pilot, not a method efficacy conclusion'})
    return out


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--val-cache')
    p.add_argument('--output', required=True, help='NEW run directory; existing directories are refused')
    p.add_argument('--semantic', choices=['mean','adaptive'], default='mean')
    p.add_argument('--spatial', choices=['identity','geometry','conditioned'], default='geometry')
    p.add_argument('--steps', type=int, default=100, help='Actual optimizer updates, not sampled batches')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--semantic-strength', type=float, default=1.0)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume')
    return p

if __name__ == '__main__':
    run(parser().parse_args())
