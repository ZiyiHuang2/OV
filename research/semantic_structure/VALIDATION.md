# v2 execution record — 2026-10-28

Parent reviewed: 3542de23d8a14b5e4b8db590ab2c5d9dc41f8361. Its all-ignore optimizer-step fix is preserved and strengthened by regression testing with populated AdamW momentum state.

Executed locally with Python 3.12, torch 2.14.1+cpu and torchvision 0.29.1+cpu:

- `python -m compileall -q .`: passed.
- `python -m unittest -v test_model test_extract`: 12 tests passed (4.531 seconds in recorded combined run).
- Actual `run_pair.py` CLI: synthetic train/source-validation caches, spatial intervention, 2 updates per arm, width 4; initial weights and sample traces matched.
- Actual `evaluate.py` CLI: saved fixed checkpoint on a disjoint-ID synthetic research cache; 3 per-image confusion matrices and JSON result produced.

Covered: common normalization; singleton-description equivalence; parameter and sampling parity; class permutation/subset and batch equivalence; masked padding; finite gradients across six modes; ignored-pixel zero direct gradient; all-ignore no optimizer/weight changes; border-safe transport; semantic coefficient zero equivalence; nonzero condition effect; strict cache/split/overlap rejection; hand-calculated metrics; exact continuous-vs-resumed CPU training; exact checkpoint reload; local CLIP checkpoint conversion; strict ResNet loading; RGB/mask preprocessing; projected dense features; text length rejection.

Image-adapter tests used temporary RGB/mask files and explicitly synthetic CLIP/ResNet weights, never described as pretrained or real remote-sensing evidence. Adapter component execution is verified; full pretrained extraction CLI with real BPE assets/checkpoints must still be checked on the server. No real dataset, pretrained weight, GPU experiment, official HG reproduction, or scientific efficacy result was available here. No artifacts containing synthetic weights/cache/predictions are committed.

Scientific scope: corrected computational interventions support controlled pilots. They do not establish that semantic ambiguity or structure propagation caused the historical DLRSD unseen degradation. ImageNet ResNet features cannot stand in for RSIE/DINO without declaring the change. One description per class cannot test semantic selection. Single-seed short-budget gains cannot support broad claims.
