#!/usr/bin/env python3
"""Compare original default computations against the pre-prototype OV revision."""
import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config
from cat_seg.engine.checkpoint import model_config
from cat_seg.modeling.transformer.model import Aggregator

REFERENCE = 'eb470ec81ef27ea2df9e2bdc559f69fb3cba39bc'


def read_reference(ref, path):
    result = subprocess.run(['git', 'show', f'{ref}:{path}'], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout
    url = f'https://raw.githubusercontent.com/ZiyiHuang2/OV/{ref}/{path}'
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode('utf-8')


def compare(ref=REFERENCE):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'ovrs_reference'
        for path in ['__init__.py', 'modeling/__init__.py', 'modeling/transformer/__init__.py', 'engine/__init__.py']:
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('')
        paths = ['modeling/transformer/model.py', 'engine/checkpoint.py']
        paths += ['modeling/modules/' + name + '.py' for name in ['__init__', 'common', 'guidance', 'fusion', 'decoder']]
        for path in paths:
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(read_reference(ref, 'cat_seg/' + path))
        spec = importlib.util.spec_from_file_location('ovrs_reference', root / '__init__.py')
        package = importlib.util.module_from_spec(spec)
        sys.modules['ovrs_reference'] = package
        spec.loader.exec_module(package)
        reference = importlib.import_module('ovrs_reference.modeling.transformer.model')
        checkpoint = importlib.import_module('ovrs_reference.engine.checkpoint')
        try:
            options = dict(text_guidance_dim=32, text_guidance_proj_dim=16,
                           appearance_guidance_dim=32, appearance_guidance_proj_dim=16,
                           decoder_dims=(32, 16), decoder_guidance_dims=(16, 8),
                           decoder_guidance_proj_dims=(8, 4), hidden_dim=32, num_layers=2,
                           nheads=4, pooling_size=(2, 2), feature_resolution=(8, 8), window_size=4, pad_len=24)
            experiments = {
                'baseline': {},
                'full': dict(aggregation='parallel_epl', context_adapter='lsk', cad_stages=(1,)),
                'full_selective': dict(aggregation='parallel_gated', context_adapter='lsk',
                                       cad_stages=(1,), cad_pooling='weighted'),
            }
            checks = {}
            for name, variant in experiments.items():
                torch.manual_seed(42)
                original = reference.Aggregator(**options, **variant).eval()
                torch.manual_seed(42)
                current = Aggregator(**options, **variant).eval()
                assert list(current.state_dict()) == list(original.state_dict())
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(current.state_dict()[key], value, rtol=0, atol=0)
                arguments = (torch.randn(2, 32, 8, 8), torch.randn(2, 3, 1, 32),
                             [torch.randn(2, 32, 8, 8), torch.randn(2, 16, 16, 16), torch.randn(2, 8, 32, 32)])
                with torch.no_grad():
                    expected, actual = original(*arguments), current(*arguments)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                cfg = get_cfg()
                add_deeplab_config(cfg)
                add_cat_seg_config(cfg)
                cfg.merge_from_file(str(ROOT / 'configs' / 'ovrs' / (name + '.yaml')))
                assert model_config(cfg) == checkpoint.model_config(cfg)
                checks[name] = dict(parameter_initialization_equal=True, parameter_keys_equal=True,
                                    forward_exact_equal=True, max_output_error=(actual - expected).abs().max().item(),
                                    default_checkpoint_signature_equal=True)
            return dict(reference_commit=ref, scope='aggregation, decoder and default checkpoint configuration; no CLIP pipeline or training-loss comparison',
                        experiments=checks)
        finally:
            for key in list(sys.modules):
                if key == 'ovrs_reference' or key.startswith('ovrs_reference.'):
                    del sys.modules[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-ref', default=REFERENCE)
    parser.add_argument('--output', default='output/prototype_validation.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = compare(args.reference_ref)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
