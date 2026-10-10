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
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(2)
    device = torch.device(args.device)
    options = dict(aggregation='parallel_epl', context_adapter='lsk', cad_stages=(1,)) if args.full else {}
    if args.selective:
        options = dict(aggregation='parallel_gated', context_adapter='lsk', cad_stages=(1,), cad_pooling='weighted')
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
    print(json.dumps({'shape': list(output.shape), 'finite': bool(torch.isfinite(output).all()),
                      'loss': loss.item(), 'seconds': time.perf_counter() - started,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                      'includes_clip': False, 'full': args.full, 'selective': args.selective}, indent=2))


if __name__ == '__main__':
    main()
