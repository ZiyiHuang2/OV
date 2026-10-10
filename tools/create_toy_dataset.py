#!/usr/bin/env python3
"""Make deterministic synthetic RGB/ID pairs for pipeline checks, not research."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def create(root):
    root = Path(root)
    if (root / 'classes.json').exists():
        raise ValueError('Toy data already exists; choose another output path')
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'labels').mkdir(parents=True, exist_ok=True)
    colors = np.array([[170, 130, 110], [75, 75, 75], [45, 135, 65]], dtype=np.uint8)
    rng = np.random.default_rng(12)
    for index in range(8):
        mask = np.full((96, 96), 2, dtype=np.uint8)
        offset = index * 3
        mask[15 + offset:45 + offset, 12:42] = 0
        mask[:, 52:61] = 1
        noise = rng.normal(0, 6, (96, 96, 3))
        image = np.clip(colors[mask].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(root / 'images' / f'{index:03}.png')
        Image.fromarray(mask).save(root / 'labels' / f'{index:03}.png')
    (root / 'classes.json').write_text(json.dumps(['building', 'road', 'vegetation']))
    (root / 'train.txt').write_text('\n'.join(f'{i:03}' for i in range(6)))
    (root / 'val.txt').write_text('006\n007\n')
    print(root.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/toy_source')
    create(parser.parse_args().output)
