"""Read only train/val of the existing version-1 manifest. Never open test data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Manifest:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.info = json.loads(self.path.read_text())
        if self.info.get('version') != 1 or self.info.get('ignore_label') != 255:
            raise ValueError('Expected version-1 manifest with ignore_label=255')
        self.train_classes = self.resolve(self.info['train_classes'])
        self.test_classes = self.resolve(self.info['test_classes'])
        # test_classes is vocabulary metadata only, NOT a test manifest/mask.
        self.names = json.loads(self.train_classes.read_text())
        self.all_names = json.loads(self.test_classes.read_text())
        self.seen = self.info['seen_class_ids']
        if self.names != [self.all_names[i] for i in self.seen] or len(set(self.seen)) != len(self.seen):
            raise ValueError('Source vocabulary mapping mismatch')
        if len(self.names) < 2 or len(set(self.all_names)) != len(self.all_names):
            raise ValueError('Invalid vocabulary')
        self.rows = {s: self.info[s] for s in ('train', 'val')}
        if any(not r for r in self.rows.values()):
            raise ValueError('Empty train/val split')
        paths = [{str(self.resolve(r['file_name'])) for r in self.rows[s]} for s in ('train', 'val')]
        if paths[0] & paths[1]:
            raise ValueError('Train/val paths overlap')
        self.identity = {'manifest': digest(self.path), 'train_classes': digest(self.train_classes),
                         'test_classes': digest(self.test_classes)}
        self._support = None

    def resolve(self, path):
        return (self.root/Path(path)).resolve()

    def read(self, split, index):
        if split not in self.rows:
            raise ValueError('Only train/val are accessible')
        row = self.rows[split][index]
        image = np.asarray(Image.open(self.resolve(row['file_name'])).convert('RGB')).copy()
        mask = np.asarray(Image.open(self.resolve(row['sem_seg_file_name']))).copy()
        if mask.ndim != 2 or image.shape[:2] != mask.shape:
            raise ValueError('Image/mask shape mismatch')
        k = len(self.names) if split == 'train' else len(self.all_names)
        if not np.isin(mask, list(range(k))+[255]).all():
            raise ValueError('Train masks must ALREADY be remapped; val uses original IDs')
        return torch.from_numpy(image).permute(2, 0, 1).float(), torch.from_numpy(mask.astype(np.int64))

    def support(self):
        if self._support is None:
            # Pixel presence from TRAIN ONLY. Never inspect val to select episodes.
            self._support = []
            for row in self.rows['train']:
                mask = np.asarray(Image.open(self.resolve(row['sem_seg_file_name'])))
                ids = set(int(x) for x in np.unique(mask) if x != 255)
                if not ids.issubset(set(range(len(self.names)))):
                    raise ValueError('Training mask not remapped to source IDs')
                self._support.append(ids)
        return self._support


def augment(image, labels, rng, size=384):
    """Shared online resize/flip; all source pixels remain represented, no crop oracle."""
    image = F.interpolate(image[None], (size, size), mode='bilinear', align_corners=False)[0]
    labels = F.interpolate(labels[None, None].float(), (size, size), mode='nearest')[0, 0].long()
    if torch.rand((), generator=rng) < .5:
        image, labels = image.flip(-1), labels.flip(-1)
    turns = int(torch.randint(4, (), generator=rng))
    return image.rot90(turns, (-2, -1)), labels.rot90(turns, (-2, -1))


def episode(data, rng):
    present = data.support()
    candidates = [i for i, ids in enumerate(present) if ids]
    if not candidates:
        raise ValueError('No valid source pixels for episode construction')
    for _ in range(100):
        q = candidates[int(torch.randint(len(candidates), (), generator=rng))]
        ids = sorted(present[q])
        held = [ids[int(torch.randint(len(ids), (), generator=rng))]]
        support = [i for i in candidates if i != q and present[i]-set(held)]
        if support:
            s = support[int(torch.randint(len(support), (), generator=rng))]
            return s, q, held
    raise ValueError('Cannot construct disjoint-image source episode')


def metrics(cm, seen):
    cm = cm.double()
    tp, gt, pred = cm.diag(), cm.sum(1), cm.sum(0)
    union = gt+pred-tp
    iou = torch.where(union > 0, tp/union.clamp_min(1), float('nan'))
    def mean(ids):
        x = iou[list(ids)]
        x = x[torch.isfinite(x)]
        return float(x.mean()) if len(x) else None
    s = mean(seen)
    u = mean([i for i in range(len(cm)) if i not in seen])
    return {'units': 'fraction', 'miou': mean(range(len(cm))), 'seen': s, 'unseen': u,
            'hiou': None if s is None or u is None else (2*s*u/(s+u) if s+u else 0.),
            'iou': [float(x) if torch.isfinite(x) else None for x in iou],
            'confusion_gt_rows_pred_columns': cm.long().tolist()}
