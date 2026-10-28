"""Strict, memory-resident cache contract for bounded pilots."""
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_cache(path, expected_split=None):
    data = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('visual', 'text', 'valid', 'structures', 'labels', 'metadata'):
        if key not in data:
            raise ValueError('Missing cache field: ' + key)
    v, t, mask, s, y, m = [data[k] for k in ('visual','text','valid','structures','labels','metadata')]
    if not isinstance(m, dict) or m.get('schema_version') != 2:
        raise ValueError('Cache schema_version=2 and complete provenance required; no implicit legacy conversion')
    required = ('dataset', 'split', 'sample_ids', 'image_sha256', 'class_names', 'descriptions', 'encoders', 'preprocessing')
    if any(key not in m for key in required):
        raise ValueError('Incomplete cache provenance')
    if m['split'] not in ('train', 'validation', 'research') or (expected_split and m['split'] != expected_split):
        raise ValueError('Unexpected split; sealed/test caches are refused by this pilot')
    if v.ndim != 4 or min(v.shape) < 1 or t.ndim != 3 or min(t.shape) < 1 or v.shape[1] != t.shape[2]:
        raise ValueError('Invalid aligned visual/text dimensions')
    n, k = len(v), len(t)
    if y.ndim != 3 or len(y) != n or y.dtype != torch.long or min(y.shape) < 1:
        raise ValueError('Invalid labels')
    if not ((y == 255) | ((y >= 0) & (y < k))).all():
        raise ValueError('Labels must be contiguous class IDs or 255')
    if mask.shape != t.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('Invalid description validity mask')
    if not isinstance(s, list) or not s:
        raise ValueError('Nonempty structure list required')
    previous = (0, 0)
    for tensor in [v, t] + s:
        if tensor.dtype != torch.float32 or tensor.requires_grad or not torch.isfinite(tensor).all():
            raise ValueError('Caches must contain detached finite float32 features')
    for tensor in s:
        if tensor.ndim != 4 or len(tensor) != n or min(tensor.shape) < 1:
            raise ValueError('Invalid structure shape')
        if any(a < b for a, b in zip(tensor.shape[-2:], previous)):
            raise ValueError('Structure pyramid must be coarse-to-fine')
        previous = tensor.shape[-2:]
    for key in ('sample_ids', 'image_sha256'):
        if not isinstance(m[key], list) or len(m[key]) != n or len(set(m[key])) != n:
            raise ValueError('Missing/duplicate sample identities: ' + key)
    if any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in m['image_sha256']):
        raise ValueError('Invalid image SHA256')
    if len(m['class_names']) != k or len(set(m['class_names'])) != k or len(m['descriptions']) != k:
        raise ValueError('Vocabulary metadata mismatch')
    for row, descriptions in zip(mask, m['descriptions']):
        if not descriptions or len(descriptions) != int(row.sum()) or any(not x.strip() for x in descriptions):
            raise ValueError('Description metadata mismatch')
        if len(set(x.strip().casefold() for x in descriptions)) != len(descriptions):
            raise ValueError('Deduplicate descriptions before encoding')
    if not isinstance(m['encoders'], dict) or not m['encoders'] or not isinstance(m['preprocessing'], dict) or not m['preprocessing']:
        raise ValueError('Encoder and preprocessing provenance required')
    return data


def check_compatible(train, other):
    a, b = train['metadata'], other['metadata']
    if set(a['image_sha256']) & set(b['image_sha256']):
        raise ValueError('Train/evaluation image-content overlap')
    if a['dataset'] == b['dataset'] and set(a['sample_ids']) & set(b['sample_ids']):
        raise ValueError('Train/evaluation sample-ID overlap')
    if a['encoders'] != b['encoders'] or a['preprocessing'] != b['preprocessing']:
        raise ValueError('Encoder/preprocessing mismatch')
    if train['visual'].shape[1] != other['visual'].shape[1] or [s.shape[1] for s in train['structures']] != [s.shape[1] for s in other['structures']]:
        raise ValueError('Feature-channel mismatch')


def batch(data, indices, device):
    return (data['visual'][indices].to(device), data['text'].to(device), data['valid'].to(device),
            [s[indices].to(device) for s in data['structures']], data['labels'][indices].to(device))
