"""Paired pixel transitions from saved evaluator RLEs, not area-only summaries."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

manifest_path, old_path, new_path, output = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
names = json.loads((manifest_path.parent / manifest['test_classes']).read_text())

def load(path):
    groups = defaultdict(list)
    for item in json.loads(path.read_text()):
        groups[item['file_name']].append(item)
    return groups

old, new = load(old_path), load(new_path)
rows = manifest['val']
assert set(old) == set(new) == {row['file_name'] for row in rows}
counts = np.zeros((len(names), 4), dtype=np.int64)

def decode(items, shape):
    pred = np.full(shape, -1, dtype=np.int16)
    for item in items:
        mask = mask_util.decode(item['segmentation']).astype(bool)
        assert mask.shape == shape and not np.any(pred[mask] != -1)
        pred[mask] = item['category_id']
    assert np.all(pred >= 0)
    return pred

for row in rows:
    target = np.asarray(Image.open(row['sem_seg_file_name']), dtype=np.int16)
    a = decode(old[row['file_name']], target.shape)
    b = decode(new[row['file_name']], target.shape)
    valid = target != manifest['ignore_label']
    for outcome, select in enumerate(((a == target) & (b == target),
                                      (a != target) & (b == target),
                                      (a == target) & (b != target),
                                      (a != target) & (b != target))):
        counts[:, outcome] += np.bincount(target[valid & select], minlength=len(names))[:len(names)]

output.write_text(json.dumps({'old': str(old_path), 'new': str(new_path),
                              'images': len(rows), 'columns': ['both_correct', 'corrected', 'regressed', 'both_wrong'],
                              'per_target_class': {name: counts[i].tolist() for i, name in enumerate(names)},
                              'total': counts.sum(axis=0).tolist()}, indent=2))
