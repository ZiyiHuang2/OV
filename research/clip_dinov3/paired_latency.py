"""Measure both inference paths on the same iSAID tiles with alternating model order."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from control import ClipProjection, DualPatchSegmenter, ManifestSplit


@torch.no_grad()
def measure(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = ManifestSplit(args.manifest, "val")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    baseline_state = torch.load(args.clip_checkpoint, map_location="cpu", weights_only=True)
    size = checkpoint["metadata"]["encoder_size"]
    baseline = ClipProjection(args.clip_weights, size).cuda().eval()
    dual = DualPatchSegmenter(args.clip_weights, args.dino_weights, size).cuda().eval()
    if dual.weight_hashes != checkpoint["metadata"]["weights"]:
        raise ValueError("Encoder weights differ from the training checkpoint")
    if baseline.weight_hashes != baseline_state["metadata"]["weights"]:
        raise ValueError("CLIP weights differ from the control checkpoint")
    baseline.projection.load_state_dict(baseline_state["projection"])
    dual.projection.load_state_dict(checkpoint["projection"])
    models = [baseline, dual]
    texts = [model.text_features(dataset.classes) for model in models]
    first = dataset[0][0].unsqueeze(0).cuda()
    for _ in range(30):
        for model, text in zip(models, texts):
            model(first, text).argmax(dim=1)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = [[], []]
    indices = np.linspace(0, len(dataset) - 1, 512, dtype=int)
    sample_ids = []
    for pair_index, index in enumerate(indices):
        image, _, sample_id = dataset[index]
        sample_ids.append(sample_id)
        image = image.unsqueeze(0).cuda()
        order = (0, 1) if pair_index % 2 == 0 else (1, 0)
        for branch in order:
            start.record()
            models[branch](image, texts[branch]).argmax(dim=1)
            end.record()
            end.synchronize()
            timings[branch].append(start.elapsed_time(end))
    result = {
        "device": torch.cuda.get_device_name(),
        "checkpoint_steps": checkpoint["steps"],
        "clip_checkpoint_steps": baseline_state["steps"],
        "batch_size": 1,
        "warmup_per_variant": 30,
        "paired_images": len(sample_ids),
        "ordered_sample_ids": sample_ids,
        "alternating_order": True,
    }
    for key, values in zip(("clip_projection", "dual"), timings):
        result[key] = {stat: float(value) for stat, value in zip(
            ("mean_ms", "p50_ms", "p95_ms"),
            (np.mean(values), np.median(values), np.percentile(values, 95)),
        )}
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(output, encoding="utf-8")
    print({name: result[name] for name in ("clip_projection", "dual")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "clip-weights", "dino-weights", "checkpoint", "clip-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    measure(parser.parse_args())
