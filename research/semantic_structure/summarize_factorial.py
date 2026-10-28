"""Export fixed-endpoint comparisons; never choose checkpoints from validation."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args(); root = Path(a.root)
    protocol = json.loads((root/'protocol.json').read_text())
    matched = json.loads((root/'matched_checks.json').read_text())
    assert matched['passed']
    rows, classes, images, curves, metrics = [], [], [], [], {}
    endpoint = protocol['training']['updates']
    reference_gt = None
    reference_ids = None
    for name,(semantic,spatial) in protocol['arms'].items():
        path = root/'training'/name
        result = json.loads((path/f'validation_{endpoint:06d}.json').read_text())
        metrics[name] = result
        summary = json.loads((path/'summary.json').read_text())
        config = json.loads((path/'protocol.json').read_text())
        log = [json.loads(line) for line in (path/'train.jsonl').read_text().splitlines()]
        assert len(log) == endpoint == summary['updates']
        assert reference_gt is None or result['gt_pixels'] == reference_gt
        ids = [r['sample_id'] for r in result['per_image']]
        assert reference_ids is None or ids == reference_ids
        reference_gt,reference_ids = result['gt_pixels'],ids
        cm = result['confusion_rows_gt_columns_prediction']
        total = sum(result['gt_pixels'])
        rows.append(dict(arm=name, semantic=semantic, spatial=spatial,
            updates=endpoint, miou_percent=100*result['miou'],
            pixel_accuracy_percent=100*sum(cm[i][i] for i in range(len(cm)))/total,
            trainable_parameters=config['parameters'],
            effective_parameters=config['parameters']-1,
            peak_allocated_mib=summary['peak_allocated_bytes']/2**20,
            mean_step_ms=statistics.mean(r['step_seconds'] for r in log)*1000,
            median_step_ms=statistics.median(r['step_seconds'] for r in log)*1000,
            elapsed_seconds=summary['elapsed_seconds_this_invocation'],
            loss_first100=statistics.mean(r['loss'] for r in log[:100]),
            loss_last100=statistics.mean(r['loss'] for r in log[-100:]),
            reload_exact=summary['reload_exact']))
        for i,label in enumerate(result['class_names']):
            gt,pred,tp=result['gt_pixels'][i],result['pred_pixels'][i],cm[i][i]
            classes.append(dict(arm=name,class_name=label,iou=result['iou'][i],
                precision=result['precision'][i],recall=result['recall'][i],gt_pixels=gt,
                predicted_pixels=pred,false_positive_pixels=pred-tp,
                false_positive_rate=(pred-tp)/(total-gt) if total>gt else None))
        for r in result['per_image']:
            images.append(dict(arm=name,sample_id=r['sample_id'],miou=r['miou'],
                               valid_pixels=sum(r['gt_pixels'])))
        for step in range(protocol['training']['eval_every'],endpoint+1,protocol['training']['eval_every']):
            r=json.loads((path/f'validation_{step:06d}.json').read_text())
            curves.append(dict(arm=name,update=step,miou_percent=100*r['miou']))
    write_csv(root/'summary.csv',rows)
    write_csv(root/'per_class.csv',classes)
    write_csv(root/'per_image.csv',images)
    write_csv(root/'learning_curve.csv',curves)
    scores={k:100*v['miou'] for k,v in metrics.items()}
    effects={a+'-'+b:scores[a]-scores[b] for a,b in [('C','B'),('D','B'),('E','B'),('E','C'),('E','D')]}
    effects['interaction_E-C-D+B']=scores['E']-scores['C']-scores['D']+scores['B']
    result={'miou_percent':scores,'effects_percentage_points':effects,
            'same_pixel_domain':True,'same_image_order':True,
            'uncertainty':'One seed and one source validation split; descriptive comparisons only',
            'effective_parameter_note':'Shared scalar head bias cancels class softmax; excluded from effective count',
            'no_unseen_or_hIoU':True}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
