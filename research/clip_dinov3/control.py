"""Train and evaluate the matched CLIP-only iSAID linear projection."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from run import CLIP_WIDTH, PROMPT, DualPatchSegmenter, ManifestSplit, evaluate, write_report


class ClipProjection(DualPatchSegmenter):
    def __init__(self, clip_weights, size):
        super().__init__(clip_weights, None, size)
        self.projection = nn.Linear(CLIP_WIDTH, CLIP_WIDTH)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(CLIP_WIDTH))
            self.projection.bias.zero_()

    def clip_patches(self, images):
        return self.clip_tokens(self.resize_rgb(images))[:, 1:]

    def forward(self, images, text):
        features = self.clip_patches(images)
        return self.pixel_logits(self.projection(features), text, images.shape[-2:])


def reconstruct_order(args, data):
    # The original run advanced the CPU RNG during model construction before shuffling.
    torch.manual_seed(args.seed)
    original = DualPatchSegmenter(args.clip_weights, args.reference_dino, args.size)
    rng_after_init = torch.get_rng_state()
    del original

    def shuffled_indices(length):
        torch.set_rng_state(rng_after_init)
        return [index for batch in DataLoader(range(length), batch_size=args.batch_size, shuffle=True)
                for index in batch.tolist()]

    if args.reference_train_audit:
        probe = json.loads(args.reference_train_audit.read_text(encoding="utf-8"))
        first_ids = [data.rows[i]["sample_id"]
                     for i in shuffled_indices(probe["train_split_size"])[:args.batch_size]]
        if first_ids != probe["audit"]["first_train_ids"]:
            raise ValueError(f"Reference shuffle replay mismatch: {first_ids}")
    return shuffled_indices(len(data))


def train(args, model, data, metadata, order):
    start = perf_counter()
    model.train()
    assert not model.clip.training and model.dino is None
    assert [name for name, param in model.named_parameters() if param.requires_grad] == [
        "projection.weight", "projection.bias",
    ]
    frozen = [(param, param._version) for param in model.clip.parameters()]
    text = model.text_features(data.classes)
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=args.lr)
    loader = DataLoader(data, batch_size=args.encode_batch_size, sampler=order,
                        num_workers=args.workers, pin_memory=True)
    steps, seen, loss_sum = 0, 0, 0.0
    digest = sha256()
    audit = {}
    for images, masks, sample_ids in loader:
        images = images.to(args.device, non_blocking=True)
        masks = masks.to(args.device, non_blocking=True)
        features = model.clip_patches(images)
        for offset in range(0, len(images), args.batch_size):
            if args.max_steps and steps >= args.max_steps:
                break
            group = slice(offset, offset + args.batch_size)
            patches, labels = features[group], masks[group]
            optimizer.zero_grad(set_to_none=True)
            logits = model.pixel_logits(model.projection(patches), text, labels.shape[-2:])
            if steps == 0:
                baseline = model.pixel_logits(patches, text, labels.shape[-2:])
                direct = model.clip_patches(images[group])
                torch.testing.assert_close(logits, baseline, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(patches, direct, atol=1e-4, rtol=1e-5)
                audit.update(
                    identity_logit_max_error=(logits - baseline).abs().max().item(),
                    encoder_batch_max_feature_error=(patches - direct).abs().max().item(),
                    first_train_ids=sample_ids[group],
                )
            loss = F.cross_entropy(logits, labels, ignore_index=data.ignore_id)
            value = loss.item()
            if not isfinite(value):
                raise ValueError(f"Nonfinite loss for {sample_ids[group]}")
            loss.backward()
            if steps == 0:
                audit["first_projection_grad_norm"] = model.projection.weight.grad.norm().item()
                assert audit["first_projection_grad_norm"] > 0
            optimizer.step()
            steps += 1
            seen += len(labels)
            loss_sum += value
            for sample_id in sample_ids[group]:
                digest.update((sample_id + "\n").encode())
            if steps == 1 or steps % 100 == 0:
                print(f"step={steps} loss={value:.6f} mean_loss={loss_sum / steps:.6f} "
                      f"elapsed_s={perf_counter() - start:.1f}", flush=True)
        if args.max_steps and steps >= args.max_steps:
            break

    assert all(param.grad is None and param._version == version for param, version in frozen)
    expected = [data.rows[i]["sample_id"] for i in order[:seen]]
    assert digest.hexdigest() == sha256("".join(i + "\n" for i in expected).encode()).hexdigest()
    checkpoint = args.output / "projection.pt"
    torch.save({"metadata": metadata, "steps": steps, "projection": model.projection.state_dict()},
               checkpoint)
    if args.max_steps:
        val = ManifestSplit(args.manifest, "val", 1)
        image, _, sample_id = val[0]
        with torch.no_grad():
            logits = model.eval()(image.unsqueeze(0).to(args.device), model.text_features(val.classes))
        torch.save({"sample_id": sample_id, "logits": logits.cpu()}, args.output / "roundtrip.pt")
    return {
        **metadata, "steps": steps, "images_with_supervision_batches": seen,
        "ordered_train_sample_ids_sha256": digest.hexdigest(),
        "mean_step_loss": loss_sum / steps, "last_loss": value, "checkpoint": str(checkpoint),
        "train_wall_seconds": perf_counter() - start,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        "audit": {**audit, "clip_parameter_tensors_frozen": len(frozen)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reference-dino", type=Path)
    parser.add_argument("--reference-train-audit", type=Path,
                        help="Optional dual run audit report used to verify the shuffle replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    if args.mode == "train" and (args.reference_dino is None or args.encode_batch_size % args.batch_size):
        parser.error("training needs the reference DINO weights and an integer number of optimizer batches")
    if args.mode == "eval" and args.checkpoint is None:
        parser.error("evaluation needs the trained CLIP projection")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split = "train" if args.mode == "train" else "val"
    data = ManifestSplit(args.manifest, split, args.limit)
    order = reconstruct_order(args, data) if args.mode == "train" else None
    torch.manual_seed(args.seed)
    model = ClipProjection(args.clip_weights, args.size).to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest": str(args.manifest), "split": split, "classes": data.classes,
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "variant": "clip-projection", "ignore_id": data.ignore_id, "encoder_size": args.size,
        "prompt": PROMPT, "weights": model.weight_hashes, "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(), "batch_size": args.batch_size,
        "dtype": "float32", "tf32": False,
        "trainable_parameters": sum(p.numel() for p in model.projection.parameters()),
    }
    if args.mode == "train":
        metadata.update(lr=args.lr, seed=args.seed, epochs=1, train_split_size=len(data),
                        encoder_batch_size=args.encode_batch_size)
        report = train(args, model, data, metadata, order)
    else:
        report = evaluate(args, model, data, metadata)
    write_report(args.output, report)


if __name__ == "__main__":
    main()
