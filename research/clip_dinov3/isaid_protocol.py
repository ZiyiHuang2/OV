"""iSAID snapshot schedule, held-class data preparation and checkpoint retention."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


HELD_CLASSES = {
    'H1': ('storage tank', 'tennis court', 'bridge'),
    'H2': ('plane', 'basketball court', 'roundabout'),
}


def development_epochs(epochs):
    if epochs == 40:
        return tuple(sorted({15, *range(2, epochs, 2)}))
    return (5, 10) if epochs == 15 else (1, 3)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare_isaid_held(source, dest, split):
    from prepare_supervision import prepare_held

    source, dest = Path(source).resolve(), Path(dest).resolve()
    with TemporaryDirectory(prefix='.held-', dir=dest.parent) as temp:
        prepared = Path(temp) / 'data'
        prepare_held(source, prepared, HELD_CLASSES[split])
        path = prepared / 'manifest.json'
        manifest = read_json(path)
        for row in manifest['train']:
            row['sem_seg_file_name'] = Path(row['sem_seg_file_name']).relative_to(prepared).as_posix()
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        prepared.rename(dest)


def held_identity(source, path, split):
    source, path = Path(source), Path(path)
    original, held = read_json(source), read_json(path)
    audit = read_json(path.parent / 'split_audit.json')
    classes = read_json(source.parent / original['train_classes'])
    seen = [name for name in classes if name not in HELD_CLASSES[split]]
    source_hash = sha256(source.read_bytes()).hexdigest()
    if (held['source_manifest_sha256'] != source_hash
            or audit['source_manifest_sha256'] != source_hash
            or set(audit['held_classes']) != set(HELD_CLASSES[split])
            or read_json(path.parent / held['train_classes']) != seen
            or read_json(path.parent / held['test_classes']) != classes
            or held['seen_class_ids'] != [classes.index(name) for name in seen]
            or held['ignore_label'] != original['ignore_label']):
        raise ValueError('Held-class manifest does not match the requested source and vocabulary')

    digest = sha256(path.read_bytes())
    for i, (a, b) in enumerate(zip(original['train'], held['train'], strict=True)):
        rel_mask = f'masks/{i:08d}.png'
        if (a['sample_id'] != b['sample_id']
                or (source.parent / a['file_name']).resolve() != Path(b['file_name'])
                or b['sem_seg_file_name'] != rel_mask
                or b.get('pixel_status_file_name')):
            raise ValueError(f'Mismatched held-class training sample: {a["sample_id"]}')
        digest.update(sha256((path.parent / rel_mask).read_bytes()).digest())
    for a, b in zip(original['val'], held['val'], strict=True):
        if (a['sample_id'] != b['sample_id']
                or any((source.parent / a[key]).resolve() != Path(b[key])
                       for key in ('file_name', 'sem_seg_file_name'))):
            raise ValueError(f'Mismatched held-class validation sample: {a["sample_id"]}')
    return digest.hexdigest()


def finish_tracked_run(dest, epochs, held):
    dest, epochs = Path(dest), int(epochs)
    train = dest / 'train'
    points = (*development_epochs(epochs), epochs)
    candidates = []
    sample_hash = None
    for epoch in points:
        report = read_json(dest / f'dev_epoch_{epoch:03d}' / 'report.json')
        if sample_hash is None:
            sample_hash = report['sample_ids_sha256']
        elif sample_hash != report['sample_ids_sha256']:
            raise ValueError('Development checkpoints were evaluated on different samples')
        # Keep the denominator fixed even when a category absent from the
        # development labels gets predicted at only some checkpoints.
        active = {name: report['class_iou'][name] for name, row in
                  zip(report['test_classes'], report['confusion_matrix'], strict=True) if sum(row)}
        if held:
            held_scores = [active[name] for name in HELD_CLASSES[held] if name in active]
            seen_scores = [score for name, score in active.items() if name in report['train_classes']]
            if not seen_scores or not held_scores:
                raise ValueError('Development split must contain both seen and held categories')
            a, b = sum(seen_scores) / len(seen_scores), sum(held_scores) / len(held_scores)
            score = 2 * a * b / (a + b) if a + b else 0.
        else:
            score = sum(active.values()) / len(active)
        candidates.append((score, epoch, report['checkpoint_steps']))
    score, epoch, steps = max(candidates, key=lambda entry: entry[0])
    best = train / 'best.pt'
    info_file = train / 'best.json'
    info = {'epoch': epoch, 'steps': steps, 'score': score, 'sample_ids_sha256': sample_hash,
            'criterion': 'gt_supported_harmonic_iou' if held else 'gt_supported_miou'}
    if not (best.is_file() and info_file.is_file() and read_json(info_file) == info):
        source = train / ('final.pt' if epoch == epochs else f'epoch_{epoch:03d}.pt')
        tmp = best.with_suffix('.tmp')
        shutil.copyfile(source, tmp)
        tmp.replace(best)
        tmp = info_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        tmp.replace(info_file)
    retained = {15, 30} if epochs == 40 else set()
    for epoch in points[:-1]:
        if epoch not in retained:
            (train / f'epoch_{epoch:03d}.pt').unlink(missing_ok=True)
