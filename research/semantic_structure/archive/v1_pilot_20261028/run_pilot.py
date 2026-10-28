"""Matched, fixed-endpoint cached-feature engineering checks and pilot."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from model import SemanticStructureDecoder, segmentation_loss

ARMS = [('B', 'mean', 'fusion'), ('C', 'adaptive', 'fusion'),
        ('D', 'mean', 'transport'), ('E', 'adaptive', 'transport')]


def tensor_hash(values):
    h = hashlib.sha256()
    for value in values:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def scores(cm):
    diag = cm.diag().double()
    gt, pred = cm.sum(1).double(), cm.sum(0).double()
    union = gt + pred - diag
    def divide(numerator, denominator):
        return [float(x / y) if y else None for x, y in zip(numerator, denominator)]
    return {'per_class_iou': divide(diag, union),
            'per_class_precision': divide(diag, pred),
            'per_class_recall': divide(diag, gt),
            'gt_pixels': gt.long().tolist(), 'predicted_pixels': pred.long().tolist(),
            'miou': float((diag[union > 0] / union[union > 0]).mean()) if (union > 0).any() else None,
            'classes_in_miou': int((union > 0).sum())}


def confusion(logits, target, classes):
    valid = target != 255
    if not ((target[valid] >= 0) & (target[valid] < classes)).all():
        raise ValueError('label outside vocabulary')
    ids = target[valid] * classes + logits.argmax(1)[valid]
    return torch.bincount(ids.cpu(), minlength=classes ** 2).reshape(classes, classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=['engineering', 'pilot'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    conf = json.loads(Path(args.protocol).read_text())
    train = torch.load(Path(args.features) / 'train.pt', weights_only=True)
    val = torch.load(Path(args.features) / 'val.pt', weights_only=True)
    for key in ('text', 'valid'):
        torch.testing.assert_close(train[key], val[key], rtol=0, atol=0)
    for split, data in [('train', train), ('val', val)]:
        if data['metadata']['classes'] != conf['class_names']:
            raise ValueError('class order differs from protocol')
        if data['metadata']['manifest_sha256'] != conf['manifest_sha256']:
            raise ValueError('manifest differs from protocol')
        if [r['index'] for r in data['metadata']['records']] != conf['indices'][split]:
            raise ValueError('sample identities differ from protocol')
        if any(r['split'] != split for r in data['metadata']['records']):
            raise ValueError('incorrect split in cache')
    n = conf['engineering']['train_first_n'] if args.phase == 'engineering' else len(train['visual'])
    settings = conf[args.phase]
    steps, batch = settings['updates'], settings['batch_size']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results, initial_hash = {}, None
    for name, semantic, spatial in ARMS:
        torch.manual_seed(conf['seed'])
        torch.cuda.manual_seed_all(conf['seed'])
        model = SemanticStructureDecoder(train['visual'].shape[1],
                    [s.shape[1] for s in train['structures']],
                    semantic_mode=semantic, spatial_mode=spatial).cuda()
        digest = tensor_hash(model.state_dict().values())
        if initial_hash is None:
            initial_hash = digest
        assert digest == initial_hash
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf['pilot']['lr'])
        generator = torch.Generator().manual_seed(conf['seed'])
        updates, attempts, elapsed, losses = 0, 0, [], []
        text, valid = train['text'].cuda(), train['valid'].cuda()
        def forward(data, indices):
            target = data['labels'][indices].cuda()
            logits = model(data['visual'][indices].cuda(), text, valid,
                           [s[indices].cuda() for s in data['structures']], target.shape[-2:])
            return logits, target
        def subset_loss():
            model.eval()
            with torch.no_grad():
                value = sum(float(segmentation_loss(*forward(train, [i]))) for i in range(n)) / n
            model.train()
            return value
        before = subset_loss() if args.phase == 'engineering' else None
        torch.cuda.reset_peak_memory_stats()
        while updates < steps:
            indices = torch.randint(n, (batch,), generator=generator)
            attempts += 1
            if not (train['labels'][indices] != 255).any():
                if attempts > steps * 100:
                    raise RuntimeError('too many all-ignore batches')
                continue
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, target = forward(train, indices)
            loss = segmentation_loss(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
            updates += 1
            with (output / f'{name}_train.jsonl').open('a') as f:
                f.write(json.dumps({'step': updates, 'indices': indices.tolist(),
                                    'loss': losses[-1], 'seconds': elapsed[-1]}) + '\n')
        training_peak = torch.cuda.max_memory_allocated()
        effective = sum(p.numel() for p in model.parameters() if p.grad is not None)
        after = subset_loss() if args.phase == 'engineering' else None
        model.eval()
        with torch.no_grad():
            expected, _ = forward(train, [0])
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'semantic': semantic, 'spatial': spatial, 'step': updates,
                      'sample_attempts': attempts, 'rng_sampler': generator.get_state(),
                      'init_sha256': digest, 'protocol': conf, 'phase': args.phase,
                      'elapsed': elapsed, 'losses': losses}
        torch.save(checkpoint, output / f'{name}.pt')
        restored = SemanticStructureDecoder(train['visual'].shape[1],
                    [s.shape[1] for s in train['structures']],
                    semantic_mode=semantic, spatial_mode=spatial).cuda().eval()
        restored.load_state_dict(torch.load(output / f'{name}.pt', weights_only=True)['model'])
        with torch.no_grad():
            actual = restored(train['visual'][:1].cuda(), text, valid,
                              [s[:1].cuda() for s in train['structures']], train['labels'].shape[-2:])
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        cm = torch.zeros((len(text), len(text)), dtype=torch.int64)
        per_image = []
        if args.phase == 'pilot':
            with torch.no_grad():
                for i in range(len(val['visual'])):
                    logits, target = forward(val, [i])
                    c = confusion(logits, target, len(text))
                    cm += c
                    per_image.append({'id': val['metadata']['records'][i]['id'],
                                      'confusion_matrix': c.tolist(), **scores(c)})
        result = {'config': name, 'semantic': semantic, 'spatial': spatial,
                  'init_sha256': digest, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
                  'gradient_connected_parameters': effective,
                  'effective_parameters': effective - model.head.bias.numel(),
                  'parameter_note': 'Excludes inactive mode scalars and the shared head bias, which cancels in softmax',
                  'updates': updates, 'sample_attempts': attempts,
                  'loss_first': losses[0], 'loss_last': losses[-1],
                  'overfit_loss_before': before, 'overfit_loss_after': after,
                  'training_peak_bytes': training_peak,
                  'seconds_per_step_mean': sum(elapsed) / len(elapsed),
                  'reload_exact': True, 'output_shape': list(expected.shape),
                  'confusion_matrix': cm.tolist(), 'per_image': per_image}
        if args.phase == 'pilot':
            result.update(scores(cm))
        results[name] = result
        (output / f'{name}_metrics.json').write_text(json.dumps(result, indent=2))
        print(name, json.dumps({k: v for k, v in result.items()
                               if k not in ('confusion_matrix', 'per_image')}), flush=True)
        if args.phase == 'engineering' and not after < before:
            raise RuntimeError(name + ' engineering loss failed to decrease')
    (output / 'summary.json').write_text(json.dumps(results, indent=2))
    print('PASS', args.phase, flush=True)


if __name__ == '__main__':
    main()
