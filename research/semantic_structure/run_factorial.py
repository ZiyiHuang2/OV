"""Execute the explicitly authorized four-arm frozen-feature experiment.

Run extraction first. This runner gates the expanded experiment on real-data
engineering acceptance, checks matched controls, and never selects a best epoch.
"""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from data import load_cache, check_compatible, batch, sha256, write_json
from model import SemanticStructureDecoder, segmentation_loss


def subset(data, indices):
    result = {k: data[k] for k in ('text', 'valid')}
    result.update(visual=data['visual'][indices].clone(), labels=data['labels'][indices].clone(),
                  structures=[s[indices].clone() for s in data['structures']])
    m = copy.deepcopy(data['metadata'])
    for key in ('sample_ids', 'image_sha256', 'records'):
        m[key] = [m[key][i] for i in indices]
    result['metadata'] = m
    return result


def tensor_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            tensor_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            tensor_equal(x, y)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root)
    protocol = json.loads((root/'protocol.json').read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train = load_cache(root/'features/train.pt', 'train')
    val = load_cache(root/'features/val.pt', 'validation')
    check_compatible(train, val)
    assert len(train['visual']) == protocol['training']['train_count']
    assert len(val['visual']) == protocol['training']['val_count']
    assert train['metadata']['sample_ids'] == protocol['sample_ids']['train']
    assert val['metadata']['sample_ids'] == protocol['sample_ids']['val']
    small = subset(train, protocol['engineering']['indices'])
    torch.save(small, root/'engineering_cache.pt')
    write_json(root/'cache_identity.json', {key:sha256(root/path) for key,path in
        [('train','features/train.pt'),('val','features/val.pt'),('engineering','engineering_cache.pt')]})
    del train, val
    arms = protocol['arms']

    def launch(name, stage, steps, cache, gpu, resume=None):
        cfg = protocol['engineering'] if stage.startswith('engineering') else protocol['training']
        output = root/stage/name
        output.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).with_name('train_cached.py')),
               '--cache', str(cache), '--output', str(output), '--semantic', arms[name][0],
               '--spatial', arms[name][1], '--steps', str(steps), '--batch-size',str(cfg['batch_size']),
               '--width','64','--seed','42','--lr','0.0002','--weight-decay','0.01',
               '--temperature','0.1','--semantic-strength','1.0','--device','cuda:0',
               '--save-every', str(cfg['save_every']), '--eval-every',str(cfg['eval_every'])]
        if stage == 'training':
            cmd += ['--val-cache', str(root/'features/val.pt')]
        if resume:
            cmd += ['--resume',str(resume)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        write_json(output.parent/(name+'_launch.json'), {'command':cmd, 'physical_gpu':gpu,
                   'started_unix':time.time(), 'protocol_sha256':sha256(root/'protocol.json')})
        with (output.parent/(name+'.log')).open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((output/'summary.json').read_text())

    acceptance = {}
    for name in arms:
        summary = launch(name,'engineering',20,root/'engineering_cache.pt',0)
        launch(name,'engineering_part',10,root/'engineering_cache.pt',0)
        launch(name,'engineering_resume',20,root/'engineering_cache.pt',0,
               root/'engineering_part'/name/'final.pt')
        full = torch.load(root/'engineering'/name/'final.pt',weights_only=True)
        resumed = torch.load(root/'engineering_resume'/name/'final.pt',weights_only=True)
        for key in ('model','optimizer','sampler_state','sample_trace','torch_rng_state','cuda_rng_state'):
            tensor_equal(full[key],resumed[key])
        torch.manual_seed(42)
        model = SemanticStructureDecoder(**full['config']).cuda()
        inputs = batch(small,torch.arange(len(small['visual'])),'cuda')
        def loss():
            with torch.no_grad():
                v,t,m,s,y = inputs
                logits = model(v,t,m,s,y.shape[-2:])
                assert logits.shape == (4,14,224,224)
                return float(segmentation_loss(logits,y))
        before = loss()
        model.load_state_dict(full['model'],strict=True)
        after = loss()
        assert after < before, (name,before,after)
        acceptance[name] = dict(loss_before=before,loss_after=after,resume_exact=True,**summary)
        write_json(root/'engineering_acceptance.json',acceptance)
        print('ENGINEERING_PASS',name,before,after,flush=True)
        del model, inputs, full, resumed
        torch.cuda.empty_cache()
    assert len({v['initial_state_sha256'] for v in acceptance.values()}) == 1
    assert len({v['sample_trace_sha256'] for v in acceptance.values()}) == 1
    # Different processes isolate peak memory measurements. At most one per GPU.
    # The fourth arm runs on GPU 0 after B; no other user's process is touched.
    def lane(names, gpu):
        for name in names:
            launch(name,'training',protocol['training']['updates'],root/'features/train.pt',gpu)
            print('TRAINING_COMPLETE',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lane,['B','E'],0),pool.submit(lane,['C'],1),pool.submit(lane,['D'],2)]
        for future in futures:
            future.result()
    summaries = {name:json.loads((root/'training'/name/'summary.json').read_text()) for name in arms}
    protocols = {name:json.loads((root/'training'/name/'protocol.json').read_text()) for name in arms}
    for key in ('initial_state_sha256','sample_trace_sha256','updates'):
        assert len({s[key] for s in summaries.values()}) == 1, key
    for name in arms:
        assert protocols[name]['controls'] == protocols['B']['controls']
        assert protocols[name]['parameters'] == protocols['B']['parameters']
        assert summaries[name]['reload_exact']
    write_json(root/'matched_checks.json',{'passed':True,'summaries':summaries,
        'identical_controls':True,'identical_initialization':True,'identical_sampling':True,
        'checkpoint_selection':'fixed update 4000','physical_gpu_assignment':{'B':0,'C':1,'D':2,'E':0}})
    print('ALL_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
