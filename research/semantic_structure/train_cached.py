"""Small frozen-feature integration pilot; see README for cache contract."""
import argparse
from pathlib import Path
import torch
from model import SemanticStructureDecoder, segmentation_loss

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--semantic", choices=["mean", "adaptive"], default="mean")
    p.add_argument("--spatial", choices=["fusion", "transport"], default="fusion")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.steps < 1 or a.batch_size < 1:
        p.error("steps and batch size must be positive")
    destination = Path(a.output)
    if destination.exists():
        raise FileExistsError(destination)
    torch.manual_seed(a.seed)
    data = torch.load(a.cache, map_location="cpu", weights_only=True)
    if not isinstance(data.get("metadata"), dict):
        raise ValueError("Cache provenance metadata is required")
    v, t, valid, structures, labels = [data[k] for k in
        ("visual", "text", "valid", "structures", "labels")]
    if len(v) == 0 or len(labels) != len(v) or any(len(s) != len(v) for s in structures):
        raise ValueError("Empty or mismatched sample counts")
    if labels.dtype != torch.long or not ((labels == 255) | ((labels >= 0) & (labels < len(t)))).all():
        raise ValueError("Invalid labels")
    if not (labels != 255).any():
        raise ValueError("Training cache contains no supervised pixels")
    model = SemanticStructureDecoder(v.shape[1], [s.shape[1] for s in structures],
        semantic_mode=a.semantic, spatial_mode=a.spatial).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    t, valid = t.to(a.device), valid.to(a.device)
    for step in range(a.steps):
        indices = torch.randint(len(v), (min(a.batch_size, len(v)),))
        target = labels[indices].to(a.device)
        y = model(v[indices].to(a.device), t, valid,
                  [s[indices].to(a.device) for s in structures], target.shape[-2:])
        loss = segmentation_loss(y, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step == 0 or (step + 1) % 10 == 0:
            print({"step": step + 1, "train_loss": float(loss.detach())}, flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "args": vars(a), "metadata": data["metadata"],
                "clip_dim": v.shape[1], "structure_dims": [s.shape[1] for s in structures],
                "torch_rng_state": torch.get_rng_state()}, destination)

if __name__ == "__main__":
    main()
