"""Synthetic GPU contracts; these tests are not real-data effectiveness evidence."""
import io
import unittest
import torch
from model import SemanticStructureDecoder, segmentation_loss


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class GPUContracts(unittest.TestCase):
    def test_shapes_paths_gradients_and_reload(self):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        for classes in (1, 3, 5):
            torch.manual_seed(13)
            visual = torch.randn(2, 8, 3, 5, device='cuda')
            text = torch.randn(classes, 3, 8, device='cuda')
            valid = torch.arange(3, device='cuda')[None] <= torch.arange(classes, device='cuda')[:, None] % 3
            structures = [torch.randn(2, 5, 3, 5, device='cuda'), torch.randn(2, 6, 6, 10, device='cuda')]
            target = torch.randint(classes, (2, 9, 15), device='cuda')
            target[:, 0] = 255
            outputs = []
            for semantic in ('mean', 'adaptive'):
                for spatial in ('fusion', 'transport'):
                    torch.manual_seed(42)
                    model = SemanticStructureDecoder(8, [5, 6], 4, semantic, spatial).cuda()
                    output = model(visual, text, valid, structures, target.shape[-2:])
                    self.assertEqual(tuple(output.shape), (2, classes, 9, 15))
                    loss = segmentation_loss(output, target)
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        active = not ((name == 'log_temperature' and semantic == 'mean') or
                                      (name == 'semantic_strength' and spatial == 'fusion'))
                        self.assertEqual(parameter.grad is not None, active, name)
                        if active:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    stream = io.BytesIO()
                    torch.save(model.state_dict(), stream); stream.seek(0)
                    restored = SemanticStructureDecoder(8, [5, 6], 4, semantic, spatial).cuda()
                    restored.load_state_dict(torch.load(stream, weights_only=True))
                    torch.testing.assert_close(output, restored(visual, text, valid, structures, target.shape[-2:]), rtol=0, atol=0)
                    outputs.append(output.detach())
            if classes > 1:
                for i in range(4):
                    for j in range(i):
                        self.assertGreater(float((outputs[i] - outputs[j]).abs().max()), 1e-7)


if __name__ == '__main__':
    unittest.main()
