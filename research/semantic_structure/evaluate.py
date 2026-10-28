"""Fixed-checkpoint evaluation; no threshold fitting or target selection."""
import argparse
from pathlib import Path
import torch
from data import load_cache, check_compatible, batch, sha256, write_json
from model import SemanticStructureDecoder, MODEL_VERSION


def confusion(prediction, labels, classes):
    valid = labels != 255
    return torch.bincount((labels[valid]*classes + prediction[valid]).cpu(),
                          minlength=classes*classes).reshape(classes, classes)


def metrics(matrix, names):
    cm = matrix.double()
    tp, gt, pred = cm.diag(), cm.sum(1), cm.sum(0)
    union = gt + pred - tp
    def ratio(num, den):
        return [float(n/d) if d > 0 else None for n, d in zip(num, den)]
    iou = ratio(tp, union)
    present = [x for x in iou if x is not None]
    return {'units': 'fraction', 'class_names': names, 'iou': iou,
            'precision': ratio(tp, pred), 'recall': ratio(tp, gt),
            'gt_pixels': gt.long().tolist(), 'pred_pixels': pred.long().tolist(),
            'miou': sum(present)/len(present) if present else None,
            'absent_class_policy': 'exclude only classes with zero union',
            'confusion_rows_gt_columns_prediction': matrix.tolist()}


@torch.no_grad()
def evaluate(model, data, device='cpu', batch_size=1):
    was_training = model.training
    model.eval()
    total = torch.zeros(len(data['text']), len(data['text']), dtype=torch.long)
    rows = []
    try:
        for start in range(0, len(data['visual']), batch_size):
            ids = torch.arange(start, min(start+batch_size, len(data['visual'])))
            v, t, mask, structures, labels = batch(data, ids, device)
            logits = model(v, t, mask, structures, labels.shape[-2:])
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite evaluation logits')
            prediction = logits.argmax(1)
            for j, index in enumerate(ids.tolist()):
                cm = confusion(prediction[j], labels[j], len(t))
                total += cm
                rows.append({'sample_id': data['metadata']['sample_ids'][index],
                             **metrics(cm, data['metadata']['class_names'])})
    finally:
        model.train(was_training)
    return {**metrics(total, data['metadata']['class_names']), 'per_image': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('Positive batch size required')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('model_version') != MODEL_VERSION:
        raise ValueError('Incompatible checkpoint version')
    data = load_cache(a.cache)
    if data['metadata']['split'] == 'train':
        raise ValueError('Use validation/research cache for evaluation')
    provenance = saved['train_signature']
    # Construct only the shape/provenance fields used by the compatibility check.
    check_compatible({'metadata': provenance, 'visual': torch.empty(0, saved['config']['clip_dim']),
                      'structures': [torch.empty(0, d) for d in saved['config']['structure_dims']]}, data)
    model = SemanticStructureDecoder(**saved['config']).to(a.device)
    model.load_state_dict(saved['model'], strict=True)
    result = evaluate(model, data, a.device, a.batch_size)
    result.update(checkpoint_sha256=sha256(a.checkpoint), cache_sha256=sha256(a.cache),
                  checkpoint_updates=saved['updates'], config=saved['config'])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)

if __name__ == '__main__':
    main()
