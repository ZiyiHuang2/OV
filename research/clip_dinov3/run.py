"""Train or evaluate the standalone CLIP + DINOv3 projection on a segmentation manifest."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "_deps" / name) for name in ("CLIP", "dinov3")]
from model import CLIP_WIDTH, PROMPT, DualPatchSegmenter


class ManifestSplit(Dataset):
    def __init__(self, path, split, limit=None):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        class_path = manifest["train_classes" if split == "train" else "test_classes"]
        self.classes = json.loads((path.parent / class_path).read_text(encoding="utf-8"))
        self.ignore_id = manifest["ignore_label"]
        self.rows = manifest[split][:limit]
        self.root = path.parent

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["file_name"]) as source:
            image = np.array(source.convert("RGB"))
        with Image.open(self.root / row["sem_seg_file_name"]) as source:
            mask = np.array(source, dtype=np.int64)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image and mask dimensions differ for {row['sample_id']}")
        invalid = (mask != self.ignore_id) & ((mask < 0) | (mask >= len(self.classes)))
        if invalid.any():
            raise ValueError(f"Unexpected mask IDs for {row['sample_id']}: {np.unique(mask[invalid])}")
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask), row["sample_id"]


def write_report(output, report):
    data = json.dumps(report, indent=2, allow_nan=False)
    (output / "report.json").write_text(data + "\n", encoding="utf-8")
    print(data, flush=True)


def train(args, model, data, metadata):
    start = perf_counter()
    model.train()
    text = model.text_features(data.classes)
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=args.lr)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    audit = {}
    steps = 0
    valid_images = 0
    loss_sum = 0
    for epoch in range(args.epochs):
        for images, masks, sample_ids in loader:
            if not (masks != data.ignore_id).any():
                continue
            images = images.to(args.device, non_blocking=True)
            masks = masks.to(args.device, non_blocking=True)
            first = args.audit and steps == 0
            if first:
                assert not model.clip.training and not model.dino.training
                assert [name for name, param in model.named_parameters() if param.requires_grad] == [
                    "projection.weight", "projection.bias",
                ]
                frozen = [(param, param._version) for param in
                          (*model.clip.parameters(), *model.dino.parameters())]
                audit.update(model.audit_initialization(images, text))
                dino_before = model.projection.weight[:, CLIP_WIDTH:].detach().clone()

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images, text), masks, ignore_index=data.ignore_id)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite loss on {list(sample_ids)}")
            loss.backward()
            optimizer.step()
            steps += 1
            valid_images += len(images)
            loss_sum += loss.item()
            if first:
                assert all(param.grad is None and param._version == version for param, version in frozen)
                assert all(torch.isfinite(param.grad).all() for param in model.projection.parameters())
                grad_norm = model.projection.weight.grad[:, CLIP_WIDTH:].norm().item()
                update_norm = (model.projection.weight[:, CLIP_WIDTH:] - dino_before).norm().item()
                assert grad_norm > 0 and update_norm > 0
                audit.update({
                    "frozen_parameter_tensors": len(frozen),
                    "frozen_encoders_unchanged": True,
                    "dinov3_projection_grad_norm": grad_norm,
                    "dinov3_projection_update_norm": update_norm,
                    "first_train_ids": list(sample_ids),
                    "first_loss": loss.item(),
                })
            if first or steps % 100 == 0:
                print(f"step={steps} epoch={epoch + 1} loss={loss.item():.6f} "
                      f"mean_loss={loss_sum / steps:.6f} elapsed_s={perf_counter() - start:.1f}", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        if args.max_steps and steps >= args.max_steps:
            break
    if not steps:
        raise RuntimeError("Training found no pixels with a supervised label")
    checkpoint = args.output / "projection.pt"
    torch.save({"metadata": metadata, "steps": steps, "projection": model.projection.state_dict()},
               checkpoint)
    if args.audit:
        val = ManifestSplit(args.manifest, "val", 1)
        image, _, sample_id = val[0]
        model.eval()
        with torch.no_grad():
            prediction = model(image.unsqueeze(0).to(args.device), model.text_features(val.classes))
        torch.save({"sample_id": sample_id, "logits": prediction.cpu()}, args.output / "roundtrip.pt")
    report = {
        **metadata, "steps": steps, "images_with_supervision_batches": valid_images,
        "mean_step_loss": loss_sum / steps, "last_loss": loss.item(), "checkpoint": str(checkpoint),
        "train_wall_seconds": perf_counter() - start,
    }
    if args.audit:
        report["audit"] = audit
    return report


@torch.no_grad()
def evaluate(args, model, data, metadata):
    model.eval()
    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for field in ("weights", "encoder_size", "prompt", "manifest_sha256", "classes", "ignore_id"):
            if state["metadata"][field] != metadata[field]:
                raise ValueError(f"Checkpoint metadata mismatch in {field}")
        model.projection.load_state_dict(state["projection"], strict=True)
    text = model.text_features(data.classes)
    classes = len(data.classes)
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=args.device)
    reference = (torch.load(args.reference, map_location="cpu", weights_only=True)
                 if args.reference else None)
    roundtrip_error = None
    cuda = torch.device(args.device).type == "cuda"
    if cuda:
        warmup_images = torch.stack([data[i][0] for i in range(min(len(data), args.batch_size))]).to(args.device)
        for _ in range(args.warmup):
            model(warmup_images, text).argmax(dim=1)
        torch.cuda.synchronize()
        del warmup_images
        torch.cuda.reset_peak_memory_stats()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    peak_inference = 0
    sample_digest = sha256()
    evaluated_images = 0
    start = perf_counter()
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    for images, masks, sample_ids in loader:
        images = images.to(args.device, non_blocking=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            event_start.record()
        logits = model(images, text)
        prediction = logits.argmax(dim=1)
        if cuda:
            event_end.record()
            event_end.synchronize()
            times_ms.append(event_start.elapsed_time(event_end))
            peak_inference = max(peak_inference, torch.cuda.max_memory_allocated())
        if reference is not None and reference["sample_id"] in sample_ids:
            compared = logits[sample_ids.index(reference["sample_id"])].cpu().unsqueeze(0)
            torch.testing.assert_close(compared, reference["logits"], atol=1e-6, rtol=1e-6)
            roundtrip_error = (compared - reference["logits"]).abs().max().item()
        masks = masks.to(args.device, non_blocking=True)
        valid = masks != data.ignore_id
        pair = masks[valid] * classes + prediction[valid]
        confusion += torch.bincount(pair, minlength=classes * classes).reshape(classes, classes)
        if cuda:
            # Exclude the evaluator's own allocations from the next measured forward.
            del images, logits, prediction, masks, valid, pair
        for sample_id in sample_ids:
            sample_digest.update((sample_id + "\n").encode())
        evaluated_images += len(sample_ids)
        if evaluated_images % 1000 == 0:
            print(f"evaluated={evaluated_images} elapsed_s={perf_counter() - start:.1f}", flush=True)
    if reference is not None and roundtrip_error is None:
        raise ValueError(f"Round-trip reference {reference['sample_id']} was outside the evaluated rows")
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    iou = intersection[present].float() / union[present]
    return {
        **metadata, "checkpoint": str(args.checkpoint) if state else None,
        "trained_steps": state["steps"] if state else 0,
        "evaluated_images": evaluated_images, "ordered_sample_ids_sha256": sample_digest.hexdigest(),
        "valid_pixels": confusion.sum().item(), "confusion_matrix": confusion.tolist(),
        "miou_percent": 100 * iou.mean().item() if iou.numel() else None,
        "iou_percent": {data.classes[i]: 100 * (intersection[i] / union[i]).item() if present[i] else None
                        for i in range(classes)},
        "checkpoint_roundtrip_max_abs_error": roundtrip_error,
        "eval_wall_seconds": perf_counter() - start,
        "cuda_batch_ms": {
            "count": len(times_ms), "total": sum(times_ms),
            "mean": float(np.mean(times_ms)), "p50": float(np.median(times_ms)),
            "p95": float(np.percentile(times_ms, 95)), "warmup_batches": args.warmup,
        } if cuda else None,
        "peak_cuda_mib": peak_inference / 2**20 if cuda else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    eval_parser = subparsers.add_parser("eval")
    for subparser in (train_parser, eval_parser):
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--clip-weights", type=Path, required=True)
        subparser.add_argument("--dino-weights", type=Path)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--size", type=int, default=384)
        subparser.add_argument("--batch-size", type=int, default=1)
        subparser.add_argument("--limit", type=int)
        subparser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--audit", action="store_true")
    eval_parser.add_argument("--checkpoint", type=Path)
    eval_parser.add_argument("--reference", type=Path)
    eval_parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    if args.mode == "train" and args.dino_weights is None:
        parser.error("training the dual projection requires --dino-weights")
    if args.mode == "eval" and bool(args.checkpoint) != bool(args.dino_weights):
        parser.error("provide both --dino-weights and --checkpoint for dual evaluation, or neither for CLIP-only")

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if args.mode == "train":
        torch.manual_seed(args.seed)
    split = "train" if args.mode == "train" else "val"
    data = ManifestSplit(args.manifest, split, args.limit)
    if not len(data):
        raise ValueError(f"Empty {split} split")
    args.output.mkdir(parents=True, exist_ok=True)
    dino_weights = args.dino_weights.expanduser() if args.dino_weights else None
    model = DualPatchSegmenter(args.clip_weights.expanduser(), dino_weights, args.size).to(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    manifest_hash = sha256(args.manifest.read_bytes()).hexdigest()
    metadata = {
        "manifest": str(args.manifest), "split": split, "classes": data.classes,
        "manifest_sha256": manifest_hash, "variant": "dual" if dino_weights else "clip-only",
        "ignore_id": data.ignore_id, "encoder_size": args.size, "prompt": PROMPT,
        "weights": model.weight_hashes, "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(device) if cuda else str(device),
        "batch_size": args.batch_size, "dtype": "float32", "tf32": False,
        "trainable_parameters": sum(param.numel() for param in model.parameters()
                                    if param.requires_grad),
    }
    if args.mode == "train":
        metadata.update(lr=args.lr, seed=args.seed, epochs=args.epochs, train_split_size=len(data))
        report = train(args, model, data, metadata)
    else:
        report = evaluate(args, model, data, metadata)
    if cuda and args.mode == "train":
        report["peak_cuda_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    write_report(args.output, report)


if __name__ == "__main__":
    main()
