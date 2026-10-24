"""Run the CRTV verifier on cached correlation/text tensors.

This is intentionally independent of Detectron2 so the first hypothesis test
can run on exported tensors without retraining or rebuilding the full model.

Example:
    python tools/crtv_tensor_diagnostic.py \
        --correlation correlation.pt --text-features text_features.pt \
        --output crtv_diagnostics.pt --residual-scale 0.25
"""

import argparse
from pathlib import Path

import torch

from cat_seg.modeling.modules.evidence_verifier import CounterfactualEvidenceVerifier


def load_tensor(path: Path, name: str) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must contain a torch.Tensor, got {type(value)!r}")
    return value.float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlation", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relation-temperature", type=float, default=0.07)
    parser.add_argument("--margin-temperature", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=0)
    args = parser.parse_args()

    correlation = load_tensor(args.correlation, "correlation")
    text_features = load_tensor(args.text_features, "text-features")
    verifier = CounterfactualEvidenceVerifier(
        temperature=args.relation_temperature,
        margin_temperature=args.margin_temperature,
        residual_scale=args.residual_scale,
        topk=args.topk,
    ).eval()
    with torch.no_grad():
        verified, diagnostics = verifier(correlation, text_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"verified": verified, **diagnostics}, args.output)
    print(f"saved {args.output}")
    print(f"correlation: {tuple(correlation.shape)}")
    print(f"verified: {tuple(verified.shape)}")
    print(f"mean_margin: {diagnostics['margin'].mean().item():.6f}")


if __name__ == "__main__":
    main()
