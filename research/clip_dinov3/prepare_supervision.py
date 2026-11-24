"""Prepare auditable held-class or independently verified irrelevant-pixel manifests.

Input is the original run.py manifest. Never infer a negative from ignore_label.
For held-class experiments source-val remains a development set, not external test.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from PIL import Image


def dump(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')


def masks(row,root,classes,ignore):
    path=root/row['sem_seg_file_name']
    with Image.open(path) as im: y=np.array(im,dtype=np.int64)
    if y.ndim!=2 or ((y!=ignore)&((y<0)|(y>=classes))).any():
        raise ValueError(f'Invalid semantic mask: {path}')
    return y


def absolute(row,root):
    out=dict(row)
    for k in ('file_name','sem_seg_file_name','pixel_status_file_name'):
        if k in out: out[k]=str((root/out[k]).resolve())
    return out


def prepare_held(source,output,held):
    root=source.parent; m=json.loads(source.read_text(encoding='utf-8'))
    train=json.loads((root/m['train_classes']).read_text());test=json.loads((root/m['test_classes']).read_text())
    if train!=test or len(set(train))!=len(train): raise ValueError('Require same original train/val vocabulary')
    if not held or not set(held)<set(train): raise ValueError('Held names must be a nonempty proper subset')
    if m['ignore_label']!=255 or len(train)>255: raise ValueError('This PNG remapper requires ignore=255 and <=255 classes')
    ignore=255; seen=[c for c in train if c not in held]; old_to_new=np.full(256,ignore,dtype=np.uint8)
    for i,c in enumerate(seen): old_to_new[train.index(c)]=i
    output.mkdir(parents=True,exist_ok=False); (output/'masks').mkdir()
    rows=[]; counts=np.zeros(len(train),dtype=np.int64); val_counts=counts.copy(); names=set()
    for i,row in enumerate(m['train']):
        if row['sample_id'] in names: raise ValueError('Duplicate train sample')
        names.add(row['sample_id']); y=masks(row,root,len(train),ignore)
        counts+=np.bincount(y[y!=ignore],minlength=len(train))
        dest=output/'masks'/f'{i:08d}.png'; Image.fromarray(old_to_new[y]).save(dest)
        r=absolute(row,root); r['sem_seg_file_name']=str(dest.resolve())
        # Never use earlier OOV verification after changing taxonomy.
        for key in ('pixel_status_file_name','pixel_status_sha256','pixel_status_source'): r.pop(key,None)
        rows.append(r)
    vals=[]; val_names=set()
    for row in m['val']:
        if row['sample_id'] in names or row['sample_id'] in val_names: raise ValueError('Train/val sample overlap or duplicate')
        val_names.add(row['sample_id']); y=masks(row,root,len(test),ignore); val_counts+=np.bincount(y[y!=ignore],minlength=len(test))
        vals.append(absolute(row,root))
    if any(counts[train.index(c)]==0 for c in seen) or any(val_counts[train.index(c)]==0 for c in held):
        raise ValueError('Some seen training or held validation classes have no pixels')
    dump(output/'train_classes.json',seen);dump(output/'test_classes.json',test)
    # Do not expose withheld names or withheld annotation counts in training vocabulary.
    out={**m,'train_classes':'train_classes.json','test_classes':'test_classes.json','train':rows,'val':vals,
         'seen_class_ids':[test.index(c) for c in seen],'split_mode':'held-class development (source val; not external test)',
         'source_manifest_sha256':sha256(source.read_bytes()).hexdigest()}
    dump(output/'manifest.json',out)
    dump(output/'split_audit.json',{'seen_classes':seen,'held_classes':held,'train_original_pixels':dict(zip(train,counts.tolist())),
         'val_pixels':dict(zip(test,val_counts.tolist())),'held_train_pixels_remapped_to_ignore':sum(int(counts[train.index(c)]) for c in held),
         'source_manifest_sha256':out['source_manifest_sha256']})


def prepare_negatives(source,output,evidence,provenance):
    """Evidence is an explicit JSON mapping sample_id -> binary independently reviewed PNG.
    Value 1 = certified not any training category, value 0 = no claim. Not a pseudo-label generator.
    """
    if not provenance.strip(): raise ValueError('Supply review/annotation source and vocabulary scope')
    root=source.parent;m=json.loads(source.read_text(encoding='utf-8')); classes=json.loads((root/m['train_classes']).read_text())
    proof=json.loads(evidence.read_text()); ignore=m['ignore_label']; known={r['sample_id'] for r in m['train']}
    if not proof or set(proof)-known: raise ValueError('Evidence must refer exclusively to training samples')
    output.mkdir(parents=True,exist_ok=False);(output/'status').mkdir();rows=[];total=0;summary=[]
    for i,row in enumerate(m['train']):
        r=absolute(row,root);y=masks(row,root,len(classes),ignore); sid=row['sample_id']
        for key in ('pixel_status_file_name','pixel_status_sha256','pixel_status_source'): r.pop(key,None)
        if sid in proof:
            ep=evidence.parent/proof[sid]
            with Image.open(ep) as im: neg=np.array(im)
            if neg.shape!=y.shape or not np.isin(neg,[0,1]).all() or ((neg==1)&(y!=ignore)).any():
                raise ValueError(f'Evidence is not binary, aligned, and disjoint from labeled positives: {sid}')
            status=(y!=ignore).astype(np.uint8);status[neg==1]=2
            dest=output/'status'/f'{i:08d}.png';Image.fromarray(status).save(dest)
            r.update(pixel_status_file_name=str(dest.resolve()),pixel_status_sha256=sha256(dest.read_bytes()).hexdigest(),pixel_status_source=provenance)
            n=int((neg==1).sum());total+=n;summary.append({'sample_id':sid,'negative_pixels':n,'evidence_sha256':sha256(ep.read_bytes()).hexdigest()})
        rows.append(r)
    if not total: raise ValueError('No verified negative pixels in supplied evidence')
    out={**m,'train_classes':str((root/m['train_classes']).resolve()),'test_classes':str((root/m['test_classes']).resolve()),'train':rows,'val':[absolute(r,root) for r in m['val']], 'source_manifest_sha256':sha256(source.read_bytes()).hexdigest()}
    dump(output/'manifest.json',out);dump(output/'status_audit.json',{'source':provenance,'vocabulary':classes,'evidence_manifest_sha256':sha256(evidence.read_bytes()).hexdigest(),'verified_negative_pixels':total,'evidence':summary})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('held','negatives'));p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--held-class',action='append',default=[]);p.add_argument('--evidence',type=Path);p.add_argument('--provenance')
    a=p.parse_args()
    if a.mode=='held': prepare_held(a.manifest,a.output,a.held_class)
    else:
        if a.evidence is None or not a.provenance:p.error('negatives requires --evidence and --provenance')
        prepare_negatives(a.manifest,a.output,a.evidence,a.provenance)
if __name__=='__main__':main()
