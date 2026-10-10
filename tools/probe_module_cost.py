#!/usr/bin/env python3
"""Count executed Conv2d MACs only; this is not a latency/memory benchmark."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cat_seg.modeling.modules import ClassGuidanceRefiner, CostGuidanceRefiner, ParallelFusion


def measure(module, *arguments):
    macs = 0
    def count(layer, inputs, output):
        nonlocal macs
        macs += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
    handles = [layer.register_forward_hook(count) for layer in module.modules()
               if isinstance(layer, torch.nn.Conv2d)]
    try:
        with torch.no_grad():
            result = module(*arguments)
    finally:
        for handle in handles:
            handle.remove()
    return dict(parameters=sum(p.numel() for p in module.parameters()),
                trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad),
                conv2d_macs=macs, output_shape=list(result.shape))


def probe(batch=1, classes=17, channels=128, grid=24, layers=2):
    spatial = torch.randn(batch, channels, classes, grid, grid)
    category = torch.randn_like(spatial)
    fusion = {}
    for name, mode, routing in [('simple', 'simple', 'learned'), ('epl_learned', 'epl', 'learned'),
                                ('epl_uniform', 'epl', 'uniform'), ('gated', 'gated', 'learned')]:
        measured = measure(ParallelFusion(channels, mode=mode, epl_routing=routing), spatial, category)
        measured['all_layers_conv2d_macs'] = layers * measured['conv2d_macs']
        fusion[name] = measured
    # Exact ViT-B first-decoder CAD dimensions, independent of fusion channels.
    cost = torch.randn(batch * classes, 96, 2 * grid, 2 * grid)
    guidance = torch.randn(batch, 32, 2 * grid, 2 * grid)
    cad = {name: measure(refiner(96, 32), cost, guidance)
           for name, refiner in [('shared_mean', CostGuidanceRefiner), ('per_class', ClassGuidanceRefiner)]}
    return dict(scope='executed Conv2d only; no CLIP, attention, normalization, activations, or decoder upsampling',
                counts_bias_additions=False, mac_is_multiply_accumulate=True,
                batch=batch, classes=classes, channels=channels, grid=grid, aggregation_layers=layers,
                cad_cost_channels=96, cad_guidance_channels=32, fusion=fusion, cad=cad,
                gpu_memory_measured=False, latency_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--classes', type=int, default=17)
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--grid', type=int, default=24)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--output', default='output/prototype_costs.json')
    args = parser.parse_args()
    if min(args.batch, args.classes, args.channels, args.grid, args.layers) < 1:
        parser.error('all dimensions must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    result = probe(args.batch, args.classes, args.channels, args.grid, args.layers)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
