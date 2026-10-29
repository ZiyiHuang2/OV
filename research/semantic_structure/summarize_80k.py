"""Export the predetermined six checkpoints; fail if matched controls differ."""
import argparse,csv,hashlib,json,statistics,subprocess,sys
from pathlib import Path

def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);r=Path(p.parse_args().root)
    rows=[];classes=[];curves=[];traces=[];rates=[];inits=[]
    for arm in 'BCDEFGH':
        d=r/'runs'/arm;s=json.loads((d/'summary.json').read_text());c=json.loads((d/'controls.json').read_text())
        assert s['updates']==80000
        inits.append(s['initial_sha']);traces.append(s['sample_trace_sha256'])
        lrhash=hashlib.sha256();losses=[]
        with (d/'train.jsonl').open() as f:
            for line in f:
                x=json.loads(line);lrhash.update(repr(x['lr']).encode()+b'\n');losses.append(x['loss'])
        assert len(losses)==80000
        rates.append(lrhash.hexdigest())
        for step in (4000,10000,20000,40000,60000,80000):
            a=json.loads((d/f'validation14_{step:06d}.json').read_text());b=json.loads((d/f'validation17_{step:06d}.json').read_text())
            curves.append(dict(arm=arm,step=step,loss_mean_last100=statistics.mean(losses[step-100:step]),seen14=a['miou'],miou17=b['miou'],seen17=b['seen_miou'],unseen17=b['unseen_miou'],hiou17=b['hiou']))
        rows.append(dict(arm=arm,semantic=c['config']['semantic_mode'],spatial=c['config']['spatial_mode'],template_index=c['template_index'],seen14=a['miou'],miou17=b['miou'],seen17=b['seen_miou'],unseen17=b['unseen_miou'],hiou17=b['hiou'],parameters=c['parameters'],peak_allocated_mib=s['peak_allocated_bytes']/2**20,elapsed_s=s['elapsed_seconds']))
        for k,name in enumerate(b['class_names']):
            classes.append(dict(arm=arm,class_name=name,iou=b['iou'][k],precision=b['precision'][k],recall=b['recall'][k],gt_pixels=b['gt_pixels'][k],pred_pixels=b['pred_pixels'][k]))
    assert len(set(inits))==len(set(traces))==len(set(rates))==1
    csvout(r/'summary.csv',rows);csvout(r/'per_class_17.csv',classes);csvout(r/'learning_curve.csv',curves)
    (r/'matched_checks.json').write_text(json.dumps({'passed':True,'initial_hash':inits[0],'sampler_hash':traces[0],'lr_sequence_hash':rates[0]},indent=2)+'\n')
    print(json.dumps(rows,indent=2))
    subprocess.run([sys.executable,str(Path(__file__).with_name('report_80k.py')),'--root',str(r)],check=True)

if __name__=='__main__':main()
