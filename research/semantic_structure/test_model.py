import unittest
import torch
from model import SemanticStructureDecoder, segmentation_loss

class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.v = torch.randn(2, 8, 3, 4)
        self.t = torch.randn(3, 2, 8)
        self.valid = torch.tensor([[True, True], [True, False], [True, True]])
        self.s = [torch.randn(2, 5, 3, 4), torch.randn(2, 6, 6, 8)]
        self.m = SemanticStructureDecoder(8, [5, 6], width=4)

    def test_class_order_and_extension(self):
        y = self.m(self.v, self.t, self.valid, self.s)
        order = torch.tensor([2, 0, 1])
        z = self.m(self.v, self.t[order], self.valid[order], self.s)
        torch.testing.assert_close(z, y[:, order])
        z = self.m(self.v, self.t[:2], self.valid[:2], self.s)
        torch.testing.assert_close(z, y[:, :2])

    def test_padding_and_empty(self):
        y = self.m(self.v, self.t, self.valid, self.s)
        t = self.t.clone(); t[1, 1] = 100
        torch.testing.assert_close(y, self.m(self.v, t, self.valid, self.s))
        bad = self.valid.clone(); bad[0] = False
        with self.assertRaises(ValueError):
            self.m(self.v, self.t, bad, self.s)

    def test_constant_transport_at_borders(self):
        h = torch.ones(2, 3, 4, 3, 4)
        actual = self.m.transport(h, torch.randn(2, 4, 3, 4), torch.randn_like(h))
        torch.testing.assert_close(actual, h)

    def test_ablation_training_and_ignore(self):
        target = torch.randint(0, 3, (2, 6, 8)); target[:, 0] = 255
        for semantic in ("mean", "adaptive"):
            for spatial in ("fusion", "transport"):
                m = SemanticStructureDecoder(8, [5, 6], 4, semantic, spatial)
                optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
                y = m(self.v, self.t, self.valid, self.s)
                y.retain_grad()
                loss = segmentation_loss(y, target); loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(y.grad[:, :, 0].abs().sum().item(), 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                optimizer.step()
                y = m(self.v, self.t, self.valid, self.s)
                loss = segmentation_loss(y, torch.full_like(target, 255))
                self.assertEqual(loss.item(), 0)
                loss.backward()

if __name__ == "__main__":
    unittest.main()
