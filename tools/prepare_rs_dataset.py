#!/usr/bin/env python3
"""Prepare explicit ID/RGB semantic masks and a portable dataset manifest.

Class names and label mappings are supplied by the user. No DLRSD class
ordering or official split is guessed. Requires only numpy and Pillow.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_classes(path):
    names = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(names, list) or not 1 <= len(names) < 255:
        raise ValueError('Classes must be a JSON list with 1..254 names')
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('Every class name must be a nonempty string')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate class names are not allowed')
    return names


def decode_mask(path, classes, label_map=None, color_map=None):
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        if mask.shape[2] != 3 or color_map is None:
            raise ValueError(f'{path}: RGB masks require an explicit --color-map')
        result = np.full(mask.shape[:2], -1, dtype=np.int32)
        for color, target in color_map.items():
            rgb = tuple(int(value) for value in color.split(','))
            if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
                raise ValueError(f'Invalid color: {color}')
            result[np.all(mask == rgb, axis=-1)] = int(target)
        if np.any(result < 0):
            colors = np.unique(mask[result < 0].reshape(-1, 3), axis=0)
            raise ValueError(f'{path}: unmapped RGB colors: {colors[:10].tolist()}')
    elif mask.ndim == 2:
        result = mask.astype(np.int32)
        if label_map is not None:
            result = np.full(mask.shape, -1, dtype=np.int32)
            result[mask == 255] = 255
            for source, target in label_map.items():
                result[mask == int(source)] = int(target)
            if np.any(result < 0):
                raise ValueError(f'{path}: unmapped IDs: {np.unique(mask[result < 0]).tolist()}')
    else:
        raise ValueError(f'{path}: expected a 2-D ID mask or 3-channel RGB mask')
    valid = (result >= 0) & (result < classes) | (result == 255)
    if not valid.all():
        raise ValueError(f'{path}: invalid target IDs: {np.unique(result[~valid]).tolist()}')
    return result.astype(np.uint8)


def index_files(directory, suffixes):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f'Missing directory: {root}')
    indexed = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in suffixes:
            key = path.relative_to(root).with_suffix('').as_posix()
            if key in indexed:
                raise ValueError(f'Duplicate relative stem: {key}')
            indexed[key] = path
    return indexed


def read_split(path, available):
    keys = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        key = line.strip()
        if not key or key.startswith('#'):
            continue
        if key not in available and Path(key).suffix:
            key = Path(key).with_suffix('').as_posix()
        if key not in available:
            raise ValueError(f'{path}: unknown image key {key}')
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f'{path}: repeated image keys')
    return keys


def prepare(args):
    names = read_classes(args.classes)
    images = index_files(args.image_dir, {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'})
    masks = index_files(args.mask_dir, {'.png', '.tif', '.tiff', '.bmp'})
    if len(images) < 2:
        raise ValueError('At least two images are required')
    missing = sorted(set(images) - set(masks))
    if missing:
        raise ValueError(f'Missing masks: {missing[:10]}')
    if bool(args.train_list) != bool(args.val_list):
        raise ValueError('Supply both --train-list and --val-list, or neither')
    if args.train_list:
        train_keys = read_split(args.train_list, images)
        val_keys = read_split(args.val_list, images)
        split_mode = 'explicit'
    else:
        if not 0 < args.val_fraction < 1:
            raise ValueError('--val-fraction must be between 0 and 1')
        keys = sorted(images)
        indices = np.random.default_rng(args.seed).permutation(len(keys))
        count = max(1, min(len(keys) - 1, round(len(keys) * args.val_fraction)))
        val_keys = sorted(keys[index] for index in indices[:count])
        train_keys = sorted(keys[index] for index in indices[count:])
        split_mode = 'random_nonofficial'
    if not train_keys or not val_keys or set(train_keys) & set(val_keys):
        raise ValueError('Train/validation splits must be nonempty and disjoint')
    seen = list(range(len(names))) if args.seen_ids is None else [
        int(value) for value in args.seen_ids.split(',')
    ]
    if not seen or len(set(seen)) != len(seen) or any(i < 0 or i >= len(names) for i in seen):
        raise ValueError('--seen-ids must be unique valid full-vocabulary IDs')
    label_map = json.loads(Path(args.label_map).read_text()) if args.label_map else None
    color_map = json.loads(Path(args.color_map).read_text()) if args.color_map else None
    if label_map is not None and color_map is not None:
        raise ValueError('Use --label-map for ID masks OR --color-map for RGB masks')
    output = Path(args.output).resolve()
    if (output / 'manifest.json').exists():
        raise ValueError('An existing manifest will not be overwritten; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    remap = np.full(256, 255, dtype=np.uint8)
    for train_id, full_id in enumerate(seen):
        remap[full_id] = train_id
    records = {'train': [], 'val': []}
    skipped = []
    observed_train_ids = set()
    for split, keys in [('train', train_keys), ('val', val_keys)]:
        for key in keys:
            mask = decode_mask(masks[key], len(names), label_map, color_map)
            with Image.open(images[key]) as image:
                width, height = image.size
            if mask.shape != (height, width):
                raise ValueError(f'{key}: image and mask dimensions differ')
            if split == 'train':
                mask = remap[mask]
                valid_ids = np.unique(mask[mask != 255])
                if not len(valid_ids):
                    skipped.append(key)
                    continue
                observed_train_ids.update(valid_ids.tolist())
            relative = Path('masks') / split / (key + '.png')
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(destination)
            records[split].append({
                'file_name': str(images[key]), 'sem_seg_file_name': relative.as_posix(),
                'height': height, 'width': width,
            })
    if not records['train']:
        raise ValueError('No training pixels remain after seen-class remapping')
    (output / 'classes_train.json').write_text(json.dumps([names[i] for i in seen], ensure_ascii=False, indent=2))
    (output / 'classes_test.json').write_text(json.dumps(names, ensure_ascii=False, indent=2))
    manifest = {
        'version': 1, 'ignore_label': 255,
        'train_classes': 'classes_train.json', 'test_classes': 'classes_test.json',
        'seen_class_ids': seen, 'split_mode': split_mode, 'seed': args.seed,
        'skipped_all_ignore_training_images': skipped,
        'training_classes_without_pixels': [names[seen[i]] for i in range(len(seen)) if i not in observed_train_ids],
        **records,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'manifest': str(output / 'manifest.json'),
                      'train_images': len(records['train']), 'val_images': len(records['val']),
                      'seen_ids': seen, 'split_mode': split_mode,
                      'training_classes_without_pixels': manifest['training_classes_without_pixels']}, ensure_ascii=False))
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--image-dir', required=True)
    result.add_argument('--mask-dir', required=True)
    result.add_argument('--classes', required=True, help='JSON list in full mask-ID order')
    result.add_argument('--output', required=True)
    result.add_argument('--train-list')
    result.add_argument('--val-list')
    result.add_argument('--val-fraction', type=float, default=0.2)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--seen-ids', help='Comma-separated full-vocabulary IDs; others become train ignore')
    result.add_argument('--label-map', help='JSON object mapping source integer IDs to full IDs/255')
    result.add_argument('--color-map', help='JSON object mapping R,G,B strings to full IDs/255')
    return result


if __name__ == '__main__':
    argument_parser = parser()
    arguments = argument_parser.parse_args()
    try:
        prepare(arguments)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        argument_parser.error(str(error))
