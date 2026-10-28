"""Run ONE preselected intervention against mean+geometry; never auto-search."""
import argparse
import json
from pathlib import Path
from train_cached import parser as train_parser, run
from data import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--val-cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--intervention',choices=['semantic','spatial'],required=True)
    p.add_argument('--hypothesis',required=True,help='State predicted failure-group improvement before execution')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    root=Path(a.output); root.mkdir(parents=True,exist_ok=False)
    write_json(root/'experiment.json',vars(a))
    variants=[('baseline','mean','geometry'),('intervention',
              'adaptive' if a.intervention=='semantic' else 'mean',
              'conditioned' if a.intervention=='spatial' else 'geometry')]
    summaries=[]; protocols=[]; metrics=[]
    for name,semantic,spatial in variants:
        args=train_parser().parse_args(['--cache',a.cache,'--val-cache',a.val_cache,
            '--output',str(root/name),'--semantic',semantic,'--spatial',spatial,
            '--steps',str(a.steps),'--batch-size',str(a.batch_size),'--width',str(a.width),
            '--seed',str(a.seed),'--device',a.device,'--eval-every',str(a.steps)])
        run(args)
        summaries.append(json.loads((root/name/'summary.json').read_text()))
        protocols.append(json.loads((root/name/'protocol.json').read_text()))
        metrics.append(json.loads((root/name/f'validation_{a.steps:06d}.json').read_text()))
    for key in ('initial_state_sha256','sample_trace_sha256','updates'):
        if summaries[0][key]!=summaries[1][key]:
            raise RuntimeError('Unmatched comparison: '+key)
    if protocols[0]['controls']!=protocols[1]['controls'] or protocols[0]['parameters']!=protocols[1]['parameters']:
        raise RuntimeError('Unmatched inputs, optimization or parameter count')
    left,right=metrics
    delta=None if left['miou'] is None or right['miou'] is None else right['miou']-left['miou']
    write_json(root/'comparison.json',{'hypothesis':a.hypothesis,'intervention':a.intervention,
        'matched_controls':True,'metric_units':'fraction','baseline_miou':left['miou'],
        'intervention_miou':right['miou'],'delta_miou':delta,
        'per_class_delta_iou':[None if x is None or y is None else y-x for x,y in zip(left['iou'],right['iou'])],
        'class_names':left['class_names'],
        'interpretation':'Single-seed source-validation pilot; no causal-root or cross-domain efficacy claim.'})

if __name__=='__main__': main()
