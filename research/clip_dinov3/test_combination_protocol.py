"""CPU integration tests for data provenance and exact optimizer/memory resume.
Synthetic data/encoders only; does not certify real weight extraction or CUDA.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch import nn
from combination_core import CombinationCore, Config
from prepare_supervision import prepare_held, prepare_negatives


def read_runner():
    run=ModuleType('run');run.ManifestSplit=Dataset;run.PROMPT='test {}'
    model=ModuleType('combination_model');model.CSeriesSegmenter=nn.Module
    spec=importlib.util.spec_from_file_location('test_runner',Path(__file__).with_name('run_combinations.py'))
    m=importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules',{'run':run,'combination_model':model}): spec.loader.exec_module(m)
    return m


class FixtureData(Dataset):
    classes=['a','b','c','d','e'];ignore_id=255
    def __len__(self):return 6
    def __getitem__(self,i):
        gen=torch.Generator().manual_seed(100+i);image=torch.randn(3,12,16,generator=gen)
        target=torch.randint(0,2,(12,16),generator=gen);target[:4]=255
        return image,target,(target!=255).long(),str(i)


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__();self.config=Config('C6',topk=1);self.core=CombinationCore(self.config,5,8,12,12,3,2)
    def text_features(self,classes):return torch.randn(len(classes),8,generator=torch.Generator().manual_seed(5))
    def forward(self,images,text,labels=None,ignore=255,status=None,pseudo_enabled=True,force_keep_all=False):
        # Image-dependent deterministic frozen substitute; no fake foundation-weight claims.
        x=images.mean(1).reshape(len(images),12,16).mean(-1)
        c=x[...,None]*torch.arange(1,9);d=x[...,None]*torch.arange(1,13)
        return self.core(c,d,text,20.,(3,4),(12,16),guides=((d,d),(d,d)),labels=labels,ignore=ignore,status=status,pseudo_enabled=pseudo_enabled,force_keep_all=force_keep_all)


class TestProtocol(unittest.TestCase):
    def test_resume_exact(self):
        runner=read_runner();data=FixtureData()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=root/'full';b=root/'resume';a.mkdir();b.mkdir()
            def args(out,steps):return SimpleNamespace(output=out,max_steps=steps,epochs=2,seed=42,batch_size=1,workers=0,device='cpu',lr=.001,weight_decay=.01,alignment_weight=.2,aux_weight=.2,pseudo_weight=.05,pseudo_warmup=0,audit=False,save_every=1,lr_schedule='cosine',train_warmup_steps=1,min_lr_ratio=.1,grad_clip=1.)
            from dataclasses import asdict
            meta=dict(weights={},size=48,prompt='test {}',manifest='fixture',train_classes=data.classes,config=asdict(Config('C6',topk=1)),seed=42,batch_size=1,lr=.001,weight_decay=.01,ignore_id=255,limit=None,alignment_weight=.2,aux_weight=.2,pseudo_weight=.05,pseudo_warmup=0,schema=2)
            meta['training_policy']=runner.training_policy(args(a,8),data)
            torch.manual_seed(123);full=FixtureModel();runner.train(args(a,8),full,data,meta,None)
            torch.manual_seed(123);part=FixtureModel();runner.train(args(b,5),part,data,meta,None)
            state=torch.load(b/'state.pt',weights_only=True);resumed=FixtureModel();runner.train(args(b,8),resumed,data,meta,state)
            aa=torch.load(a/'state.pt',weights_only=True);bb=torch.load(b/'state.pt',weights_only=True)
            for k,v in aa['core'].items(): torch.testing.assert_close(v,bb['core'][k],rtol=0,atol=0,msg=k)
            for k,v in aa['optimizer']['state'].items():
                for field,value in v.items():torch.testing.assert_close(value,bb['optimizer']['state'][k][field],rtol=0,atol=0)
            torch.testing.assert_close(aa['rng_cpu'],bb['rng_cpu'],rtol=0,atol=0)
            self.assertEqual((aa['steps'],aa['cursor'],aa['seen']),(bb['steps'],bb['cursor'],bb['seen']))
            self.assertEqual(aa['progress'],bb['progress'])
            self.assertTrue((a/'epoch_001.pt').exists() and (b/'epoch_001.pt').exists())
            for root in (a,b):
                stage=torch.load(root/'epoch_001.pt',weights_only=True)
                self.assertEqual(stage['checkpoint_kind'],'evaluation_only')
                self.assertEqual(stage['epoch'],1)
                self.assertIn('memory.bank',stage['core'])
                self.assertNotIn('optimizer',stage)
                self.assertNotIn('rng_cpu',stage)
                self.assertNotIn('progress',stage)
                self.assertFalse((root/'final.pt').exists())
            self.assertEqual(aa['optimizer']['param_groups'],bb['optimizer']['param_groups'])
    def test_lightweight_final_export(self):
        runner=read_runner(); data=FixtureData()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td); model=FixtureModel()
            from dataclasses import asdict
            args=SimpleNamespace(output=out,max_steps=None,epochs=1,seed=42,batch_size=2,workers=0,device='cpu',lr=.001,weight_decay=.01,alignment_weight=.2,aux_weight=.2,pseudo_weight=.05,pseudo_warmup=0,audit=False,save_every=500,lr_schedule='cosine',train_warmup_steps=0,min_lr_ratio=.1,grad_clip=1.)
            meta=dict(weights={},size=48,prompt='test {}',manifest='fixture',train_classes=data.classes,config=asdict(model.config),seed=42,batch_size=2,lr=.001,weight_decay=.01,ignore_id=255,limit=None,alignment_weight=.2,aux_weight=.2,pseudo_weight=.05,pseudo_warmup=0,schema=2,training_policy=runner.training_policy(args,data))
            runner.train(args,model,data,meta,None)
            final=torch.load(out/'final.pt',weights_only=True)
            state=torch.load(out/'state.pt',weights_only=True)
            self.assertEqual(set(final),{'metadata','core','epoch','steps','seen','checkpoint_kind'})
            self.assertEqual(final['checkpoint_kind'],'evaluation_only')
            for k,v in final['core'].items():
                torch.testing.assert_close(v,state['core'][k],rtol=0,atol=0,msg=k)
            self.assertLess((out/'final.pt').stat().st_size,(out/'state.pt').stat().st_size)
            self.assertFalse(list(out.glob('epoch_*.pt')))
            with self.assertRaisesRegex(ValueError,'Evaluation-only'):
                runner.train(args,FixtureModel(),data,meta,final)
    def test_schedule(self):
        runner=read_runner(); p={'planned_steps':10,'warmup_steps':2,'schedule':'cosine','min_lr_ratio':.1}
        values=[runner.step_lr(.001,i,p) for i in range(10)]
        self.assertAlmostEqual(values[0],.0005); self.assertAlmostEqual(values[1],.001)
        self.assertAlmostEqual(values[2],.001); self.assertAlmostEqual(values[-1],.0001)
        self.assertTrue(all(x>=y for x,y in zip(values[2:],values[3:])))
    def test_formal_schedule(self):
        runner = read_runner()
        for batch_size, epochs, planned_steps in ((8, 5, 11300), (24, 15, 11310)):
            with self.subTest(batch_size=batch_size, epochs=epochs):
                args = SimpleNamespace(batch_size=batch_size, epochs=epochs, lr_schedule='cosine',
                                       train_warmup_steps=300, min_lr_ratio=.1, grad_clip=1.)
                policy = runner.training_policy(args, range(18076))
                self.assertEqual(policy['planned_steps'], planned_steps)
                self.assertEqual(policy['warmup_steps'], 300)
                for step, lr in ((0, 1e-3/300), (299, 1e-3), (300, 1e-3), (planned_steps-1, 1e-4)):
                    self.assertAlmostEqual(runner.step_lr(1e-3, step, policy), lr)
    def test_fifteen_epoch_checkpoints(self):
        runner = read_runner()
        data = FixtureData()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            args = SimpleNamespace(output=out, max_steps=None, epochs=15, seed=42, batch_size=6,
                                   workers=0, device='cpu', lr=.001, weight_decay=.01,
                                   alignment_weight=.2, aux_weight=.2, pseudo_weight=.05, pseudo_warmup=0,
                                   audit=False, save_every=500, lr_schedule='cosine', train_warmup_steps=1,
                                   min_lr_ratio=.1, grad_clip=1.)
            meta = {'training_policy': runner.training_policy(args, data)}
            result = runner.train(args, FixtureModel(), data, meta, None)
            self.assertEqual(result['completed_epochs'], 15)
            self.assertEqual({p.name for p in out.glob('*.pt')},
                             {'epoch_005.pt', 'epoch_010.pt', 'final.pt', 'state.pt'})
            for name, epoch in (('epoch_005.pt', 5), ('epoch_010.pt', 10), ('final.pt', 15)):
                export = torch.load(out/name, weights_only=True)
                self.assertEqual((export['epoch'], export['steps'], export['checkpoint_kind']),
                                 (epoch, epoch, 'evaluation_only'))
    def fixture(self,root):
        (root/'classes.json').write_text(json.dumps(['a','b','c']))
        y=np.array([[0,1,2,255],[0,1,2,255]],dtype=np.uint8)
        for n in ('train','val'):Image.fromarray(y).save(root/f'{n}.png')
        Image.fromarray(np.zeros((2,4,3),dtype=np.uint8)).save(root/'rgb.png')
        m={'train_classes':'classes.json','test_classes':'classes.json','ignore_label':255,'train':[{'sample_id':'t','file_name':'rgb.png','sem_seg_file_name':'train.png'}],'val':[{'sample_id':'v','file_name':'rgb.png','sem_seg_file_name':'val.png'}]}
        p=root/'manifest.json';p.write_text(json.dumps(m));return p
    def test_held_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=self.fixture(root);out=root/'held';prepare_held(p,out,['b'])
            m=json.loads((out/'manifest.json').read_text());self.assertEqual(json.loads((out/'train_classes.json').read_text()),['a','c']);self.assertEqual(json.loads((out/'test_classes.json').read_text()),['a','b','c'])
            with Image.open(m['train'][0]['sem_seg_file_name']) as im:self.assertEqual(np.array(im).tolist(),[[0,255,1,255],[0,255,1,255]])
            with Image.open(m['val'][0]['sem_seg_file_name']) as im:self.assertEqual(np.array(im).tolist(),[[0,1,2,255],[0,1,2,255]])
            self.assertNotIn('pixel_status_file_name',m['train'][0]);self.assertEqual(json.loads((out/'split_audit.json').read_text())['held_train_pixels_remapped_to_ignore'],2)
    def test_negative_evidence_requires_independent_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=self.fixture(root);e=root/'evidence.json';e.write_text(json.dumps({'t':'proof.png'}))
            neg=np.zeros((2,4),dtype=np.uint8);neg[:,3]=1;Image.fromarray(neg).save(root/'proof.png')
            out=root/'verified';prepare_negatives(p,out,e,'synthetic reviewed fixture; a,b,c')
            m=json.loads((out/'manifest.json').read_text());r=m['train'][0]
            with Image.open(r['pixel_status_file_name']) as im:self.assertEqual(np.array(im).tolist(),[[1,1,1,2],[1,1,1,2]])
            self.assertTrue(r['pixel_status_sha256']);self.assertEqual(json.loads((out/'status_audit.json').read_text())['verified_negative_pixels'],2)
            neg[:,0]=1;Image.fromarray(neg).save(root/'proof.png')
            with self.assertRaises(ValueError):prepare_negatives(p,root/'invalid',e,'fixture')
            e.write_text(json.dumps({'v':'proof.png'}))
            with self.assertRaises(ValueError):prepare_negatives(p,root/'val_invalid',e,'fixture')

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
