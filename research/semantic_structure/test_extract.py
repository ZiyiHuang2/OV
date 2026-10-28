"""Image-adapter tests use explicitly synthetic weights, not pretrained evidence."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from extract_features import import_file, load_clip, load_structure, prepare_pair, encode_image_pair, tokens_for

class ExtractionContracts(unittest.TestCase):
    def test_image_and_strict_weight_adapters(self):
        torch.set_num_threads(1)
        from torchvision.models import resnet50
        root = Path(__file__).resolve().parents[2]
        module = import_file('fixture_clip',root/'cat_seg/third_party/model_vpt.py')
        torch.manual_seed(3)
        clip = module.CLIP(16,32,1,64,8,8,32,64,1,1).eval()
        # Synthetic constructor leaves some text tensors uninitialized; initialize explicitly.
        for p in clip.parameters():
            torch.nn.init.normal_(p, std=0.02)
        state = dict(clip.state_dict())
        for key in list(state):
            if key.endswith('attn.q_proj_weight'):
                base = key[:-len('q_proj_weight')]
                state[base+'in_proj_weight'] = torch.cat([state.pop(base+x+'_proj_weight') for x in ('q','k','v')])
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); torch.save(state,d/'clip.pt')
            rebuilt = load_clip(d/'clip.pt','state_dict',root)
            for key,value in clip.state_dict().items():
                torch.testing.assert_close(rebuilt.state_dict()[key],value,rtol=0,atol=0)
            torch.save(resnet50(weights=None).state_dict(),d/'structure.pt')
            structure = load_structure(d/'structure.pt')
            rng = np.random.default_rng(1)
            Image.fromarray(rng.integers(0,256,(24,40,3),dtype=np.uint8)).save(d/'image.png')
            Image.fromarray(np.zeros((24,40),dtype=np.uint8)).save(d/'mask.png')
            image, label = prepare_pair(d/'image.png',d/'mask.png',32,{'0':0})
            self.assertEqual(label.shape,(24,40))
            with torch.no_grad():
                v,s = encode_image_pair(rebuilt,structure,image)
                text = rebuilt.encode_text(torch.tensor([[1,2,31,0,0,0,0,0]]))
            self.assertEqual(v.shape,(1,16,4,4)); self.assertEqual(text.shape,(1,16))
            self.assertEqual(s[0].shape,(1,1024,2,2)); self.assertEqual(s[1].shape,(1,512,4,4))
            self.assertTrue(all(torch.isfinite(x).all() for x in [v,text]+s))
            self.assertTrue(all(not p.requires_grad for p in rebuilt.parameters()))
            self.assertTrue(all(not p.requires_grad for p in structure.parameters()))
            with self.assertRaises(ValueError): prepare_pair(d/'image.png',d/'mask.png',32,{'1':0})
            del state['visual.proj']; torch.save(state,d/'broken.pt')
            with self.assertRaises(ValueError): load_clip(d/'broken.pt','state_dict',root)

    def test_no_silent_text_truncation(self):
        class Tokenizer:
            encoder={'<|startoftext|>':1,'<|endoftext|>':9}
            def encode(self,text): return [2]*len(text)
        torch.testing.assert_close(tokens_for(['a'],Tokenizer(),4),torch.tensor([[1,2,9,0]]))
        with self.assertRaises(ValueError): tokens_for(['abcdef'],Tokenizer(),4)

if __name__ == '__main__': unittest.main()
