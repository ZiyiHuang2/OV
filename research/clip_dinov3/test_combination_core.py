import unittest
import torch
from combination_core import Config, CombinationCore, segmentation_loss, RelationPropagation, ContextCostFusion, CategoryRouter, pool_regions, unknown_consistency

class TestCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.b, self.grid, self.classes = 2, (3, 4), 5
        self.c = torch.randn(2, 12, 8)
        self.d = torch.randn(2, 12, 12)
        self.t = torch.randn(5, 8)
        self.cl = tuple(torch.randn(2, 12, 12) for _ in range(3))
        self.dl = tuple(torch.randn(2, 12, 12) for _ in range(2))
        self.guides = [(self.cl[1], self.dl[1]), (self.cl[0], self.dl[0])]
        self.y = torch.randint(0, 5, (2, 12, 16)); self.y[:, 0] = 255
    def model(self, v, **kwargs):
        return CombinationCore(Config(v, **kwargs), 5, 8, 12, 12, 3, 2)
    def fwd(self, m, text=None, labels=None, **kw):
        return m(self.c, None if m.cfg.variant == 'C0' else self.d, self.t if text is None else text,
                 20., self.grid, (12,16), self.cl,self.dl,self.guides,labels=labels, **kw)
    def test_all_variants(self):
        for v in ('C0','B0','D0','C1','C2','C3','C4','C5','C6'):
            with self.subTest(v=v):
                m = self.model(v); opt = torch.optim.AdamW(m.parameters(),lr=.001)
                for _ in range(3):
                    opt.zero_grad(); r = self.fwd(m,labels=self.y)
                    loss = segmentation_loss(r['logits'],self.y,pixel=m.cfg.use_pixel)
                    if 'alignment_loss' in r: loss = loss + .2*r['alignment_loss']
                    if 'aux_logits' in r: loss = loss+.2*segmentation_loss(r['aux_logits'], self.y)
                    self.assertTrue(torch.isfinite(loss)); loss.backward(); opt.step()
                    if m.cfg.use_rs: m.memory.update(r['regions'],r['region_ids'])
                self.assertEqual(tuple(r['logits'].shape),(2,5,12,16))
                for name,p in m.named_parameters():
                    self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
                m.eval(); t=torch.randn(7,8)
                r=self.fwd(m,text=t,force_keep_all=True)
                self.assertEqual(r['logits'].shape[1],7)
                perm=torch.randperm(7); r2=self.fwd(m,text=t[perm],force_keep_all=True)
                torch.testing.assert_close(r2['logits'],r['logits'][:,perm],atol=2e-5,rtol=2e-5)
                with self.assertRaises(ValueError): self.fwd(m,labels=self.y)
    def test_initial_c1_matches_b0(self):
        a=self.model('B0').eval(); b=self.model('C1').eval()
        torch.testing.assert_close(self.fwd(a)['logits'],self.fwd(b)['logits'])
    def test_actual_grid(self):
        torch.manual_seed(2); m=CombinationCore(Config('C4'),15).eval()
        c=torch.randn(1,576,512); d=torch.randn(1,576,768); t=torch.randn(15,512)
        g=[(torch.randn(1,576,768),d),(torch.randn(1,576,768),d)]
        with torch.no_grad(): r=m(c,d,t,100.,(24,24),(384,384),guides=g)
        self.assertEqual(r['logits'].shape,(1,15,384,384)); self.assertTrue(torch.isfinite(r['logits']).all())
    def test_ignore_has_no_gradient(self):
        x=torch.randn(2,5,12,16,requires_grad=True)
        segmentation_loss(x,self.y,pixel=True).backward()
        self.assertEqual(x.grad[:,:,0].abs().sum().item(),0)
    def test_explicit_negative_only(self):
        x=torch.randn(2,5,12,16,requires_grad=True); s=(self.y!=255).long(); s[:,0,0]=2
        segmentation_loss(x,self.y,pixel=True,status=s).backward()
        self.assertGreater(x.grad[:,:,0,0].abs().sum().item(),0)
        self.assertEqual(x.grad[:,:,0,1:].abs().sum().item(),0)
    def test_relation_identity_one_neighbor(self):
        m=RelationPropagation('dual',1); x=torch.randn(2,12,5)
        torch.testing.assert_close(m(x,self.d,x),x)
    def test_prune(self):
        m=self.model('C6',topk=1).eval(); t=torch.cat((self.t[:1], -self.t[:1].expand(8,-1)),0)
        # Force homogeneous visual evidence and tied duplicate negative concepts.
        self.c=self.t[:1].expand(2,12,-1).clone()
        with torch.no_grad():
            r=self.fwd(m,text=t); self.assertIn(0,r['kept'].tolist()); self.assertLess(len(r['kept']),9); rejected=torch.tensor([i for i in range(9) if i not in r['kept'].tolist()]); self.assertTrue((r['logits'][:,rejected]==-1e4).all())
            q=self.fwd(m,text=t,force_keep_all=True); self.assertEqual(len(q['kept']),9)
        self.assertEqual(m.memory.counts.sum(),0)
    def test_c5_c6_keep_all(self):
        a=self.model('C5').eval(); b=self.model('C6').eval(); missing,unexpected=b.load_state_dict(a.state_dict(),strict=False)
        self.assertFalse(unexpected); self.assertTrue(all(k.startswith('selector.') for k in missing))
        torch.testing.assert_close(self.fwd(a)['logits'],self.fwd(b,force_keep_all=True)['logits'])

    def test_components_and_small_objects(self):
        y=torch.full((1,8,8),255);y[0,0,0]=0;y[0,7,7]=0;y[0,3,3]=1
        d=torch.arange(4*12,dtype=torch.float32).reshape(1,4,12)
        r,ids=pool_regions(d,y,(2,2),2,255,"component")
        self.assertEqual(ids.tolist(),[0,0,1]);torch.testing.assert_close(r[0],d[0,0]);torch.testing.assert_close(r[1],d[0,3])
        rc,ic=pool_regions(d,y,(2,2),2,255,"class");self.assertEqual(ic.tolist(),[0,1]);torch.testing.assert_close(rc[0],(d[0,0]+d[0,3])/2)
    def test_context_gate_uses_visual_text_and_neighborhood(self):
        m=ContextCostFusion(8,12).eval();a=torch.randn(2,12,5);b=torch.randn(2,12,5);t=torch.randn(5,12)
        original=m(self.c,self.d,self.t,t,a,b,self.grid)
        for changed in (m(self.c+torch.randn_like(self.c),self.d,self.t,t,a,b,self.grid),m(self.c,self.d,self.t.flip(0),t,a,b,self.grid),m(self.c,self.d,self.t,t,a.flip(1),b,self.grid)):
            self.assertGreater(float((changed-original).abs().max().detach()),1e-6)
        original.mean().backward()
        for key,param in m.named_parameters(): self.assertGreater(float(param.grad.abs().sum()),0,key)
    def test_relation_chunks_and_crop(self):
        torch.manual_seed(9);x=torch.randn(2,12,5);ref=torch.randn_like(x);fused=torch.randn_like(x)
        a=RelationPropagation('dual',4,'joint','conditional',3);b=RelationPropagation('dual',4,'joint','conditional',128);b.load_state_dict(a.state_dict())
        full=a(x,self.d,ref,fused);torch.testing.assert_close(full,b(x,self.d,ref,fused));torch.testing.assert_close(full[..., [0,3]],a(x[...,[0,3]],self.d,ref,fused))
        self.assertGreater(float((full-a(x,self.d,ref,fused.flip(1))).abs().max().detach()),1e-6)
        full.square().mean().backward()
        for n,p in a.named_parameters(): self.assertGreater(float(p.grad.abs().sum()),0,n)
    def test_sparse_matches_dense(self):
        x=torch.randn(2,12,5);r=torch.randn_like(x);m=RelationPropagation('dual',4,'native','global',5)
        d=torch.nn.functional.normalize(self.d,dim=-1);s=torch.nn.functional.normalize(r-r.mean(-1,keepdim=True),dim=-1)
        a=.5*(d@d.transpose(1,2)+s@s.transpose(1,2));eye=torch.eye(12,dtype=torch.bool)[None];a=torch.where(eye,torch.ones_like(a),a)
        idx=a.masked_fill(eye,2.).topk(4,-1).indices;sel=torch.zeros_like(a,dtype=torch.bool).scatter_(-1,idx,True)
        expected=x+m.strength.sigmoid()*((a/.1).masked_fill(~sel,-torch.inf).softmax(-1)@x-x)
        torch.testing.assert_close(m(x,self.d,r),expected)
    def test_dual_router_retains_branch_only_evidence(self):
        c=torch.full((1,12,9),-5.);d=c.clone();c[...,0]=10;d[...,2]=10;f=c.clone()
        self.assertEqual(CategoryRouter.select(f,c,d,1,'dual').tolist(),[0,2])
        self.assertEqual(CategoryRouter.select(f,c,d,1,'union').tolist(),[0])
    def test_pseudo_unknown_abstention_and_gradient(self):
        c=torch.full((1,4,3),-8.);c[...,0]=8;d=c.clone();d[:,3]=torch.tensor([-8.,8.,-8.]); y=torch.full((1,4,4),255);status=torch.zeros_like(y);status[:,0:2,0:2]=2;status[:,0:2,2:4]=1;y[:,0:2,2:4]=0
        t,mask=unknown_consistency(c,d,(2,2),y,status,.9,.2)
        self.assertEqual(mask.tolist(),[[False,False,True,False]])
        student=torch.randn_like(c,requires_grad=True);torch.nn.functional.kl_div(student[mask].log_softmax(-1),t[mask],reduction='batchmean').backward()
        self.assertEqual(float(student.grad[~mask].abs().sum()),0);self.assertGreater(float(student.grad[mask].abs().sum()),0)
    def test_shared_decoder_and_controls(self):
        for cfg in (dict(decoder='shared'),dict(fusion='mean'),dict(fusion='separate'),dict(fusion='learned'),dict(affinity='visual'),dict(affinity='semantic'),dict(relation_source='fused'),dict(relation_control='global'),dict(region_mode='class')):
            m=self.model('C4',**cfg);r=self.fwd(m,labels=self.y);(segmentation_loss(r['logits'],self.y)+.2*r['alignment_loss']).backward();self.assertTrue(torch.isfinite(r['logits']).all())
    def test_c6_training_keeps_gt_and_negative_provenance(self):
        m=self.model('C6',topk=1);y=torch.zeros_like(self.y);y[:,0]=255;status=(y!=255).long();status[:,0,0]=2
        for _ in range(4):
            r=self.fwd(m,labels=y,status=status);self.assertEqual(len(r['kept']),5)
        for _ in range(4):
            r=self.fwd(m,labels=y);self.assertIn(0,r['kept'].tolist());self.assertGreaterEqual(len(r['kept']),2)
    def test_prediction_module_perturbation(self):
        m=self.model('C5');opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for _ in range(4):
            opt.zero_grad();r=self.fwd(m,labels=self.y);loss=segmentation_loss(r['logits'],self.y,pixel=True)+.2*r['alignment_loss']+.05*r['pseudo_loss'];loss.backward();opt.step();m.memory.update(r['regions'],r['region_ids'])
        m.eval();base=self.fwd(m)['logits'].detach();
        for module in (m.text_map,m.fusion,m.relation,m.memory.attn,m.decoder):
            p=next(module.parameters());old=p.detach().clone()
            with torch.no_grad():p.add_(torch.randn_like(p)*.2);changed=self.fwd(m)['logits'];p.copy_(old)
            self.assertGreater(float((base-changed).abs().max()),1e-7,str(type(module)))
    def test_c5_c6_nonempty_memory_keep_all(self):
        a=self.model('C5');r=self.fwd(a,labels=self.y);a.memory.update(r['regions'],r['region_ids']);a.eval()
        b=self.model('C6').eval();b.load_state_dict(a.state_dict(),strict=False);torch.testing.assert_close(self.fwd(a)['logits'],self.fwd(b,force_keep_all=True)['logits'])

    def test_decoder_receives_pruned_vocabulary(self):
        m=CombinationCore(Config('C6',topk=1),15).eval();c=torch.randn(1,1,512).expand(1,576,-1).clone();d=torch.randn(1,1,768).expand(1,576,-1).clone();t=torch.randn(128,512);g=((d,d),(d,d));seen=[]
        handle=m.decoder.register_forward_pre_hook(lambda mod,ins:seen.append(ins[2].shape[0]))
        with torch.no_grad():
            r=m(c,d,t,100.,(24,24),(384,384),guides=g);self.assertLess(len(r['kept']),128);self.assertEqual(seen,[len(r['kept'])]);self.assertEqual(r['logits'].shape,(1,128,384,384))
        handle.remove()
    def test_single_class_and_empty_region(self):
        m=self.model('C6').eval();out=self.fwd(m,text=self.t[:1]);self.assertTrue(torch.isfinite(out['logits']).all());self.assertEqual(len(out['kept']),1)
        r,ids=pool_regions(self.d,torch.full_like(self.y,255),self.grid,5,255);self.assertEqual(len(r),0);self.assertEqual(len(ids),0)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
