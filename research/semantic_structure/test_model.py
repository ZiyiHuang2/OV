import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
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

    def test_all_ignore_batch_skips_optimizer_step(self):
        # A mixed cache can sample only ignored labels on a particular step.
        seed = None
        for candidate in range(20):
            torch.manual_seed(candidate)
            initial = SemanticStructureDecoder(8, [5])
            if torch.randint(2, (1,)).item() == 0:
                seed = candidate
                break
        self.assertIsNotNone(seed)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.pt"
            output = Path(directory) / "output.pt"
            labels = torch.zeros(2, 3, 4, dtype=torch.long)
            labels[0] = 255
            torch.save({"visual": torch.randn(2, 8, 3, 4),
                        "text": torch.randn(2, 1, 8),
                        "valid": torch.ones(2, 1, dtype=torch.bool),
                        "structures": [torch.randn(2, 5, 3, 4)],
                        "labels": labels, "metadata": {}}, cache)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("train_cached.py")),
                                     "--cache", str(cache), "--output", str(output),
                                     "--steps", "1", "--batch-size", "1", "--seed", str(seed)],
                                    check=True, capture_output=True, text=True)
            self.assertIn("all_ignore", result.stdout)
            saved = torch.load(output, map_location="cpu", weights_only=True)
            for key, value in initial.state_dict().items():
                torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)
            self.assertEqual(len(saved["optimizer"]["state"]), 0)

if __name__ == "__main__":
    unittest.main()
