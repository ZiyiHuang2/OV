#!/usr/bin/env python3
"""Compare disabled-module computation against the pinned upstream source."""
import argparse
import importlib.util
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cat_seg.modeling.transformer.model import Aggregator

SOURCE = 'https://raw.githubusercontent.com/cvlab-kaist/CAT-Seg/6d3a188af95165147fe2f34a8237fa7d2633e784/cat_seg/modeling/transformer/model.py'


def compare(source):
    spec = importlib.util.spec_from_file_location('upstream_reference', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    options = dict(
        text_guidance_dim=32, text_guidance_proj_dim=16, appearance_guidance_dim=32,
        appearance_guidance_proj_dim=16, decoder_dims=(32, 16),
        decoder_guidance_dims=(16, 8), decoder_guidance_proj_dims=(8, 4),
        hidden_dim=32, num_layers=2, nheads=4, pooling_size=(2, 2),
        feature_resolution=(8, 8), window_size=4, pad_len=0,
    )
    torch.manual_seed(42)
    original = reference.Aggregator(**options).eval()
    torch.manual_seed(42)
    current = Aggregator(**options).eval()
    assert original.state_dict().keys() == current.state_dict().keys()
    current.load_state_dict(original.state_dict(), strict=True)
    arguments = (torch.randn(2, 32, 8, 8), torch.randn(2, 3, 1, 32),
                 [torch.randn(2, 32, 8, 8), torch.randn(2, 16, 16, 16),
                  torch.randn(2, 8, 32, 32)])
    with torch.no_grad():
        expected, actual = original(*arguments), current(*arguments)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    print(json.dumps({'exact_equal': True, 'max_error': (actual - expected).abs().max().item(),
                      'scope': 'aggregation and decoder, modules disabled; not training-loss equivalence'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-model', help='Local pinned upstream model.py; otherwise download it')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.upstream_model:
        compare(Path(args.upstream_model))
    else:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.py'
            urllib.request.urlretrieve(SOURCE, path)
            compare(path)
