import unittest
import torch
from combination_core import Config, CombinationCore, segmentation_loss, RelationPropagation

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
            r=self.fwd(m,text=t); self.assertEqual(r['kept'].tolist(),[0]); self.assertTrue((r['logits'][:,1:]==-1e4).all())
            q=self.fwd(m,text=t,force_keep_all=True); self.assertEqual(len(q['kept']),9)
        self.assertEqual(m.memory.counts.sum(),0)
    def test_c5_c6_keep_all(self):
        a=self.model('C5').eval(); b=self.model('C6').eval(); missing,unexpected=b.load_state_dict(a.state_dict(),strict=False)
        self.assertFalse(unexpected); self.assertTrue(all(k.startswith('selector.') for k in missing))
        torch.testing.assert_close(self.fwd(a)['logits'],self.fwd(b,force_keep_all=True)['logits'])

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
