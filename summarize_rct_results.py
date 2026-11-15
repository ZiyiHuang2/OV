"""Summarize complete source-dev evaluator confusion matrices for five arms."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

root, class_file, output = map(Path, sys.argv[1:])
names = json.loads(class_file.read_text())
rows, per_class, confusions = [], {}, {}
for arm in ('A', 'C', 'M', 'V', 'R'):
    paths = list((root / ('explore_' + arm) / 'inference').glob('event_iter_*/confusion_matrix.npy'))
    assert len(paths) == 1, (arm, paths)
    matrix = np.load(paths[0])[:-1, :-1].astype(np.float64)
    tp = np.diag(matrix)
    gt, pred = matrix.sum(axis=0), matrix.sum(axis=1)
    valid = gt > 0
    iou = np.divide(tp, gt + pred - tp, out=np.zeros_like(tp), where=(gt + pred - tp) > 0)
    f1 = np.divide(2*tp, gt + pred, out=np.zeros_like(tp), where=(gt + pred) > 0)
    rows.append({'arm': arm, 'source_mIoU': 100*iou[valid].mean(), 'source_mF1': 100*f1[valid].mean(),
                 'seen_mIoU': 100*iou[valid].mean(), 'unseen_mIoU': '', 'hIoU': '', 'step': 2000})
    per_class[arm] = {name: float(iou[i]*100) if valid[i] else None for i, name in enumerate(names)}
    matrix[np.diag_indices_from(matrix)] = 0
    indices = np.argsort(matrix.ravel())[::-1][:15]
    confusions[arm] = [{'truth': names[int(i % len(names))], 'predicted': names[int(i // len(names))],
                        'pixels': int(matrix.ravel()[i])} for i in indices]
output.mkdir(exist_ok=True, parents=True)
with (output / 'results_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(output / 'per_class_iou.json').write_text(json.dumps(per_class, indent=2))
(output / 'largest_confusions.json').write_text(json.dumps(confusions, indent=2))
print(json.dumps(rows, indent=2))
