#!/usr/bin/env python3
"""Exercise the real ViT-B-sized aggregation/decoder without CLIP weights."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cat_seg.modeling.transformer.model import Aggregator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--classes', type=int, default=17)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--selective', action='store_true', help='Test gated fusion + weighted CAD + LSK')
    parser.add_argument('--interaction', action='store_true', help='Test cost LSK + heterogeneous fusion + affinity CAD')
    parser.add_argument('--epl-routing', choices=('learned', 'uniform'), default='learned')
    parser.add_argument('--cad-scope', choices=('shared', 'per_class'), default='shared')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', help='Optional JSON result path')
    args = parser.parse_args()
    if args.batch < 1 or args.classes < 1:
        parser.error('--batch and --classes must be positive')
    if sum((args.full, args.selective, args.interaction)) > 1:
        parser.error('--full, --selective and --interaction are mutually exclusive')
    if not args.full and (args.epl_routing != 'learned' or args.cad_scope != 'shared'):
        parser.error('prototype routing/scope options require --full')
    if args.device == 'cpu':
        torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    options = dict(aggregation='parallel_epl', context_adapter='lsk', cad_stages=(1,)) if args.full else {}
    if args.selective:
        options = dict(aggregation='parallel_gated', context_adapter='lsk', cad_stages=(1,), cad_pooling='weighted')
    if args.full:
        options.update(epl_routing=args.epl_routing, cad_scope=args.cad_scope)
    if args.interaction:
        options = dict(context_adapter='cost_lsk', aggregation='parallel_interaction',
                       cad_stages=(1,), cad_scope='per_class', cad_mode='affinity')
    model = Aggregator(num_layers=2, pooling_size=(2, 2), pad_len=256, **options).to(device).train()
    batch, classes = args.batch, args.classes
    image = torch.randn(batch, 512, 24, 24, device=device)
    text = torch.randn(batch, classes, 1, 512, device=device)
    guidance = [torch.randn(batch, 512, 24, 24, device=device),
                torch.randn(batch, 256, 48, 48, device=device),
                torch.randn(batch, 128, 96, 96, device=device)]
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(image, text, guidance)
    loss = output.square().mean()
    loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    finite = bool(torch.isfinite(output).all()) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    if not finite or tuple(output.shape) != (batch, classes, 96, 96):
        raise RuntimeError('aggregation/decoder smoke failed: invalid shape or non-finite output/gradients')
    result = {'shape': list(output.shape), 'finite': finite,
                      'loss': loss.item(), 'seconds': time.perf_counter() - started,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                      'includes_clip': False, 'full': args.full, 'selective': args.selective,
                      'interaction': args.interaction, 'model_options': options,
                      'epl_routing': options.get('epl_routing', 'learned'),
                      'cad_scope': options.get('cad_scope', 'shared'),
                      'cad_mode': options.get('cad_mode', 'residual'), 'seed': args.seed}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
