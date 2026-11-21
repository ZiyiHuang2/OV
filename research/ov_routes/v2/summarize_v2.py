"""Post-hoc audit of V2 vs exactly aligned V1 Zero validation files.

Never feeds ground-truth information back into selector or training. Imagewise
mIoU changes are descriptive: per-image present classes can differ.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(zero, v2):
    b=zero['per_image']; a=v2['per_image']
    if len(b)!=len(a):
        raise ValueError('Mismatched evaluation image counts')
    if any((x['index'],x['file']) != (y['index'],y['file']) for x,y in zip(b,a)):
        raise ValueError('Mismatched dataset indices/file identity; refuse comparison')
    result={'n_images':len(a), 'deltas_pp':{}, 'miou_by_strategy':{
        'zero':zero['miou'],'v2':v2['miou']}}
    for field in ('miou','seen','unseen','hiou'):
        x,y=zero.get(field),v2.get(field)
        result['deltas_pp'][field]=None if x is None or y is None else 100*(y-x)
    counts=Counter(int(z) for x in a for z in x.get('selected',[]))
    result['window_selection_counts']={str(i):counts[i] for i in range(8)}
    result['skipped_images']=sum(bool(x.get('skip',False)) for x in a)
    result['mean_host_calls']=sum(x['encoder_calls'] for x in a)/len(a)
    improved=harmed=equal=0
    for x,y in zip(b,a):
        u,v=x.get('miou'),y.get('miou')
        if u is None or v is None:continue
        if v>u+1e-12:improved+=1
        elif v<u-1e-12:harmed+=1
        else:equal+=1
    result['imagewise_miou_change']={'improved':improved,'harmed':harmed,'equal':equal,
                                     'note':'Imagewise mIoU is not additive across images.'}
    result['gt_used_for_selection']=False
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zero-json',required=True)
    p.add_argument('--v2-json',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    zero=json.loads(Path(args.zero_json).read_text())
    v2=json.loads(Path(args.v2_json).read_text())
    summary=summarize(zero,v2)
    Path(args.output).write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(summary['deltas_pp']))


if __name__=='__main__':
    main()
