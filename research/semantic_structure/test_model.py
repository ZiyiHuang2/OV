"""Synthetic contracts only: these tests do not establish pretrained performance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from model import SemanticStructureDecoder, optimization_step
from data import load_cache, check_compatible, write_json
from evaluate import confusion, metrics, evaluate
from train_cached import run, parser


def fixture(split='train'):
    generator = torch.Generator().manual_seed(19)
    def rand(*shape):
        return torch.randn(*shape, generator=generator)
    y = torch.randint(0, 3, (3,6,8), generator=generator)
    y[0] = 255
    return {'visual':rand(3,8,3,4), 'text':rand(3,2,8), 'valid':torch.tensor([[1,1],[1,0],[1,1]],dtype=torch.bool),
        'structures':[rand(3,5,3,4),rand(3,6,6,8)], 'labels':y,
        'metadata':{'schema_version':2,'dataset':'SYNTHETIC_ENGINEERING_ONLY','split':split,
            'sample_ids':[split+str(i) for i in range(3)],
            'image_sha256':[hashlib.sha256((split+str(i)).encode()).hexdigest() for i in range(3)],
            'class_names':['a','b','c'], 'descriptions':[['a1','a2'],['b1'],['c1','c2']],
            'encoders':{'source':'synthetic fixture, no pretrained encoder'},'preprocessing':{'type':'synthetic'}}}

class ModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7); torch.set_num_threads(1)
        self.data = fixture()
        self.v, self.t, self.mask, self.s = [self.data[k] for k in ('visual','text','valid','structures')]
        self.m = SemanticStructureDecoder(8,[5,6],4,'adaptive','conditioned')

    def test_class_order_extension_and_batch(self):
        y = self.m(self.v,self.t,self.mask,self.s)
        order = torch.tensor([2,0,1])
        torch.testing.assert_close(self.m(self.v,self.t[order],self.mask[order],self.s), y[:,order])
        torch.testing.assert_close(self.m(self.v,self.t[:2],self.mask[:2],self.s), y[:,:2])
        torch.testing.assert_close(self.m(self.v[:1],self.t,self.mask,[s[:1] for s in self.s]), y[:1])

    def test_loss_contract_and_flattened_equivalence(self):
        from model import segmentation_loss
        logits = torch.randn(2,3,5,7,requires_grad=True)
        labels = torch.randint(3,(2,5,7)); labels[:,0] = 255
        torch.testing.assert_close(segmentation_loss(logits, labels),
            torch.nn.functional.cross_entropy(logits, labels, ignore_index=255))
        for bad in (labels.float(), labels[:1], torch.full_like(labels,-1), torch.full_like(labels,3)):
            with self.assertRaises(ValueError): segmentation_loss(logits,bad)
        zero = segmentation_loss(logits,torch.full_like(labels,255))
        zero.backward()
        self.assertEqual(zero.item(),0)
        self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_identical_parameter_initialization(self):
        states = []
        for semantic in ('mean','adaptive'):
            for spatial in ('identity','geometry','conditioned'):
                torch.manual_seed(12)
                states.append(SemanticStructureDecoder(8,[5,6],4,semantic,spatial).state_dict())
        for state in states[1:]:
            for key in states[0]:
                torch.testing.assert_close(state[key],states[0][key],rtol=0,atol=0)

    def test_one_description_equivalence_and_norm(self):
        baseline = SemanticStructureDecoder(8,[5,6],4,'mean','conditioned')
        baseline.load_state_dict(self.m.state_dict())
        torch.testing.assert_close(baseline(self.v,self.t[:,:1],self.mask[:,:1],self.s),
                                   self.m(self.v,self.t[:,:1],self.mask[:,:1],self.s))
        for model in (baseline,self.m):
            v = torch.nn.functional.normalize(self.v,dim=1)
            text = torch.nn.functional.normalize(self.t[0],dim=-1)
            selected = model.selected_text(v,text)
            torch.testing.assert_close(selected.norm(dim=1),torch.ones_like(selected[:,0]))

    def test_mask_padding_and_bad_inputs(self):
        expected = self.m(self.v,self.t,self.mask,self.s)
        changed = self.t.clone(); changed[1,1] = float('nan')
        torch.testing.assert_close(self.m(self.v,changed,self.mask,self.s),expected)
        mask = self.mask.clone(); mask[0] = False
        with self.assertRaises(ValueError): self.m(self.v,self.t,mask,self.s)
        with self.assertRaises(ValueError): SemanticStructureDecoder(8,[5],spatial_mode='transport')

    def test_transport_boundaries_and_zero_condition(self):
        h = torch.ones(2,3,4,3,4)
        g, e = torch.randn(2,4,3,4),torch.randn_like(h)
        torch.testing.assert_close(self.m.transport(h,g,e),h)
        plain = SemanticStructureDecoder(8,[5,6],4,'mean','geometry')
        self.m.semantic_strength = 0
        values = torch.randn_like(h)
        torch.testing.assert_close(self.m.transport(values,g,e),plain.transport(values,g,e))

    def test_condition_changes_transport(self):
        h = torch.randn(2,3,4,3,4); g = torch.randn(2,4,3,4); e = torch.randn_like(h)
        a = self.m.transport(h,g,e)
        self.m.spatial_mode = 'geometry'
        b = self.m.transport(h,g,e)
        self.assertGreater((a-b).abs().max().item(),1e-5)

    def test_gradients_and_all_ignore_adamw_state(self):
        for semantic in ('mean','adaptive'):
            for spatial in ('identity','geometry','conditioned'):
                model = SemanticStructureDecoder(8,[5,6],4,semantic,spatial)
                opt = torch.optim.AdamW(model.parameters(),lr=1e-3)
                loss = optimization_step(model,opt,self.v,self.t,self.mask,self.s,self.data['labels'])
                self.assertIsNotNone(loss)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                before = copy.deepcopy(model.state_dict()); state = copy.deepcopy(opt.state_dict())
                self.assertIsNone(optimization_step(model,opt,self.v,self.t,self.mask,self.s,torch.full_like(self.data['labels'],255)))
                for key in before: torch.testing.assert_close(model.state_dict()[key],before[key],rtol=0,atol=0)
                for idx, values in state['state'].items():
                    for key,value in values.items():
                        torch.testing.assert_close(opt.state_dict()['state'][idx][key],value,rtol=0,atol=0)
        logits = self.m(self.v,self.t,self.mask,self.s)
        logits.retain_grad()
        from model import segmentation_loss
        segmentation_loss(logits,self.data['labels']).backward()
        self.assertEqual(logits.grad[0].abs().sum().item(),0)

class PipelineContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_metrics_known_counts(self):
        cm = confusion(torch.tensor([0,1,0,1]),torch.tensor([0,0,1,255]),3)
        self.assertEqual(cm.tolist(),[[1,1,0],[1,0,0],[0,0,0]])
        result = metrics(cm,['a','b','c'])
        self.assertAlmostEqual(result['miou'],1/6)
        self.assertIsNone(result['iou'][2])

    def test_cache_rejection_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.pt'
            good = fixture(); torch.save(good,path)
            load_cache(path,'train')
            bad = copy.deepcopy(good); bad['metadata']['descriptions'][0] = ['same','same']
            torch.save(bad,path)
            with self.assertRaises(ValueError): load_cache(path)
            with self.assertRaises(ValueError): check_compatible(good,good)
            val = fixture('validation'); check_compatible(good,val)
            bad = copy.deepcopy(good); bad['metadata']['split']='test'; torch.save(bad,path)
            with self.assertRaises(ValueError): load_cache(path)

    def test_paired_runs_resume_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); train = root/'train.pt'; val = root/'val.pt'
            torch.save(fixture(),train); torch.save(fixture('validation'),val)
            def execute(name, steps, resume=None, semantic='mean',spatial='geometry'):
                args = ['--cache',str(train),'--val-cache',str(val),'--output',str(root/name),
                        '--steps',str(steps),'--width','4','--eval-every','1','--save-every','1',
                        '--semantic',semantic,'--spatial',spatial]
                if resume: args += ['--resume',str(resume)]
                run(parser().parse_args(args))
                return torch.load(root/name/'final.pt',weights_only=True)
            full = execute('full',4)
            execute('part',2)
            resumed = execute('resume',4,root/'part/final.pt')
            for key in full['model']: torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
            self.assertEqual(full['sample_trace'],resumed['sample_trace'])
            self.assertTrue(all(0 not in row for row in full['sample_trace']))
            for semantic, spatial in [('adaptive','geometry'),('mean','conditioned'),('adaptive','conditioned')]:
                other=execute(semantic+spatial,4,semantic=semantic,spatial=spatial)
                self.assertEqual(full['sample_trace'],other['sample_trace'])
                self.assertEqual(full['initial_state_sha256'],other['initial_state_sha256'])
            self.assertTrue((root/'full/validation_000004.json').exists())
            with self.assertRaises(FileExistsError): execute('full',4)

if __name__ == '__main__': unittest.main()
