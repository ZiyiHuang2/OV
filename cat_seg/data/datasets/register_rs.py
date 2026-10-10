"""Register an explicitly prepared remote-sensing dataset."""
import hashlib
import json
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog


def register_rs_manifest(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or manifest.get('ignore_label') != 255:
        raise ValueError('Expected a version-1 manifest with ignore_label=255')
    train_classes = path.parent / manifest['train_classes']
    test_classes = path.parent / manifest['test_classes']
    train_names = json.loads(train_classes.read_text())
    test_names = json.loads(test_classes.read_text())
    seen_ids = manifest['seen_class_ids']
    if train_names != [test_names[index] for index in seen_ids]:
        raise ValueError('Training vocabulary does not match seen_class_ids')
    prefix = 'ovrs_' + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    for split, classes in [('train', train_names), ('val', test_names)]:
        name = prefix + '_' + split
        entries = []
        for item in manifest[split]:
            record = dict(item)
            for field in ['file_name', 'sem_seg_file_name']:
                location = Path(record[field])
                if not location.is_absolute():
                    location = path.parent / location
                if not location.is_file():
                    raise FileNotFoundError(location)
                record[field] = str(location.resolve())
            entries.append(record)
        if not entries:
            raise ValueError(f'{split}: empty dataset')
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, lambda rows=entries: [dict(row) for row in rows])
        MetadataCatalog.get(name).set(
            stuff_classes=classes, ignore_label=255, evaluator_type='sem_seg',
            seen_class_ids=seen_ids if split == 'val' else list(range(len(classes))),
        )
    return {
        'train_name': prefix + '_train', 'test_name': prefix + '_val',
        'train_classes': str(train_classes), 'test_classes': str(test_classes),
        'num_train_classes': len(train_names),
    }
