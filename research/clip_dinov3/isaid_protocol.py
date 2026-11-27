"""iSAID snapshot schedule, held-class data preparation and checkpoint retention."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory


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


def sample_ids_sha256(ids):
    return sha256(''.join(name + '\n' for name in sorted(ids)).encode()).hexdigest()


def file_sha256(path):
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_external_source(source, external):
    source, external = Path(source), Path(external)
    source_manifest, external_manifest = read_json(source), read_json(external)
    rows = external_manifest['val']
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('External validation requires unique sample IDs')
    for key in ('file_name', 'sem_seg_file_name'):
        local_files = {(source.parent / row[key]).resolve()
                       for split in ('train', 'val') for row in source_manifest[split]}
        external_files = {(external.parent / row[key]).resolve() for row in rows}
        if len(external_files) != len(rows):
            raise ValueError(f'External validation repeats files: {key}')
        if local_files & external_files:
            raise ValueError(f'External validation references files from the iSAID source: {key}')


def audit_isaid(path, report_path):
    """Validate the full source split; reuse the pixel audit while the files match."""
    from prepare_isaid import CLASSES, SOURCE_REPO, SOURCE_REVISION, SPLIT_IDS_SHA256, SPLIT_SIZES, TILE_SIZE

    path, report_path = Path(path), Path(report_path)
    manifest_bytes = path.read_bytes()
    manifest_hash = sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest['ignore_label'] != 255:
        raise ValueError('iSAID ignore label must be 255')
    if (manifest.get('source_repo', SOURCE_REPO) != SOURCE_REPO
            or manifest.get('source_revision', SOURCE_REVISION) != SOURCE_REVISION):
        raise ValueError('iSAID source does not match the pinned OVRSISBench release')
    for key in ('train_classes', 'test_classes'):
        if read_json(path.parent / manifest[key]) != CLASSES:
            raise ValueError(f'Unexpected iSAID vocabulary: {key}')

    files = {}
    stats = sha256()
    for split, size in SPLIT_SIZES.items():
        rows = manifest[split]
        ids = [row['sample_id'] for row in rows]
        if len(ids) != size or len(set(ids)) != size:
            raise ValueError(f'{split}: expected {size} unique sample IDs, found {len(ids)} rows and {len(set(ids))} IDs')
        files[split] = []
        for row in rows:
            sample_id = row['sample_id']
            pair = tuple(path.parent / row[key] for key in ('file_name', 'sem_seg_file_name'))
            if any(file.suffix != '.png' or file.stem != sample_id for file in pair):
                raise ValueError(f'{split}: image and mask paths must match PNG sample {sample_id}')
            if row.get('pixel_status_file_name'):
                raise ValueError('Use the original iSAID manifest without extra pixel supervision')
            for file in pair:
                st = file.stat()
                stats.update(f'{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}\n'.encode())
            files[split].append((sample_id, *pair))
        if sample_ids_sha256(ids) != SPLIT_IDS_SHA256[split]:
            raise ValueError(f'{split}: sample IDs differ from the published iSAID split')

    identity = {'audit_version': 1, 'source_revision': SOURCE_REVISION,
                'source_manifest_sha256': manifest_hash, 'file_stat_sha256': stats.hexdigest()}
    if report_path.is_file():
        existing = read_json(report_path)
        if all(existing.get(key) == value for key, value in identity.items()):
            return existing

    import numpy as np
    from PIL import Image

    report = {**identity, 'source_repo': SOURCE_REPO, 'classes_by_id': CLASSES, 'ignore_label': 255}
    digests = {}
    train_images = set()
    for split, pairs in files.items():
        foreground_empty = 0
        pixels = np.zeros(256, dtype=np.int64)
        digest = sha256()
        for name, image, mask in sorted(pairs):
            image_bytes, mask_bytes = image.read_bytes(), mask.read_bytes()
            with Image.open(BytesIO(image_bytes)) as im:
                im.load()
                if im.format != 'PNG' or im.mode != 'RGB' or im.size != TILE_SIZE:
                    raise ValueError(f'{split}: invalid iSAID RGB tile: {image}')
            with Image.open(BytesIO(mask_bytes)) as im:
                if im.format != 'PNG' or im.mode != 'L' or im.size != TILE_SIZE:
                    raise ValueError(f'{split}: invalid iSAID indexed mask: {mask}')
                counts = np.bincount(np.asarray(im).ravel(), minlength=256)
            if counts[len(CLASSES):255].any():
                raise ValueError(f'{split}: invalid iSAID label IDs in {mask}')
            foreground_empty += not counts[:len(CLASSES)].any()
            pixels += counts
            image_hash = sha256(image_bytes).digest()
            if split == 'train':
                train_images.add(image_hash)
            elif image_hash in train_images:
                raise ValueError(f'{split}: image duplicates a training tile: {name}')
            digest.update(image_hash)
            digest.update(sha256(mask_bytes).digest())
        if not pixels[:len(CLASSES)].all():
            raise ValueError(f'{split}: missing labels from the published iSAID vocabulary')
        digests[split] = digest.hexdigest()
        report[split] = {'images': len(pairs), 'foreground_empty': foreground_empty,
                         'pixels_by_id': {i: int(pixels[i]) for i in (*range(len(CLASSES)), 255)},
                         'sample_ids_sha256': SPLIT_IDS_SHA256[split]}
    report['data_sha256'] = sha256(''.join(digests.values()).encode()).hexdigest()
    with NamedTemporaryFile('w', dir=report_path.parent, encoding='utf-8', delete=False) as temp:
        json.dump(report, temp, indent=2)
        temp.write('\n')
    Path(temp.name).replace(report_path)
    return report


def prepare_isaid_held(source, dest, split, source_data_hash=None):
    from prepare_supervision import prepare_held

    source, dest = Path(source).resolve(), Path(dest).resolve()
    with TemporaryDirectory(prefix='.held-', dir=dest.parent) as temp:
        prepared = Path(temp) / 'data'
        prepare_held(source, prepared, HELD_CLASSES[split])
        path = prepared / 'manifest.json'
        manifest = read_json(path)
        digest = sha256()
        for row in manifest['train']:
            mask = Path(row['sem_seg_file_name'])
            digest.update(sha256(mask.read_bytes()).digest())
            row['sem_seg_file_name'] = mask.relative_to(prepared).as_posix()
        if source_data_hash is not None:
            manifest['source_data_sha256'] = source_data_hash
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        audit_path = prepared / 'split_audit.json'
        audit = read_json(audit_path)
        audit['remapped_masks_sha256'] = digest.hexdigest()
        audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
        prepared.rename(dest)


def held_identity(source, path, split, source_data_hash=None):
    source, path = Path(source), Path(path)
    original, held = read_json(source), read_json(path)
    audit = read_json(path.parent / 'split_audit.json')
    classes = read_json(source.parent / original['train_classes'])
    seen = [name for name in classes if name not in HELD_CLASSES[split]]
    source_hash = sha256(source.read_bytes()).hexdigest()
    if (held['source_manifest_sha256'] != source_hash
            or audit['source_manifest_sha256'] != source_hash
            or held.get('source_data_sha256') != source_data_hash
            or set(audit['held_classes']) != set(HELD_CLASSES[split])
            or read_json(path.parent / held['train_classes']) != seen
            or read_json(path.parent / held['test_classes']) != classes
            or held['seen_class_ids'] != [classes.index(name) for name in seen]
            or held['ignore_label'] != original['ignore_label']):
        raise ValueError('Held-class manifest does not match the requested source and vocabulary')

    digest = sha256()
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
    masks_hash = digest.hexdigest()
    if masks_hash != audit.get('remapped_masks_sha256'):
        raise ValueError('Held-class training masks differ from the audited derivative')
    return sha256(path.read_bytes() + bytes.fromhex(masks_hash)).hexdigest()


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
        candidates.append((score, epoch, report['checkpoint_steps'], report.get('checkpoint_sha256')))
    score, epoch, steps, checkpoint_hash = max(candidates, key=lambda entry: entry[0])
    best = train / 'best.pt'
    info_file = train / 'best.json'
    info = {'epoch': epoch, 'steps': steps, 'score': score, 'sample_ids_sha256': sample_hash,
            'criterion': 'gt_supported_harmonic_iou' if held else 'gt_supported_miou'}
    if checkpoint_hash is not None:
        info['checkpoint_sha256'] = checkpoint_hash
    recorded = best.is_file() and info_file.is_file() and read_json(info_file) == info
    if not recorded or (checkpoint_hash is not None and file_sha256(best) != checkpoint_hash):
        source = train / ('final.pt' if epoch == epochs else f'epoch_{epoch:03d}.pt')
        tmp = best.with_suffix('.tmp')
        shutil.copyfile(source, tmp)
        if checkpoint_hash is not None and file_sha256(tmp) != checkpoint_hash:
            tmp.unlink()
            raise ValueError(f'Best candidate file differs from its development report: {source}')
        tmp.replace(best)
        tmp = info_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        tmp.replace(info_file)
    retained = {15, 30} if epochs == 40 else set()
    for epoch in points[:-1]:
        if epoch not in retained:
            (train / f'epoch_{epoch:03d}.pt').unlink(missing_ok=True)
