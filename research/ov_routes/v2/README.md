# Route B V2 — one exploratory run

This directory extends the existing Route B host **without changing V1** (`../host.py`, `../views.py`, `../run.py`). It is intentionally isolated from other OV research prototypes. The frozen CAT-Seg ViT-B/16 checkpoint remains the same throughout source caching, training and inference.

**Goal.** Can one spatial–semantic selector, using only first-pass GT-free evidence, choose a second RGB observation that improves open-vocabulary segmentation? This is an *exploratory experiment*, not a benchmark claim or a finished paper method. The previous DLRSD Zero/Oracle numbers are historical references and do not transfer automatically to the OVRSIS95K host/vocabulary setting.

## Frozen protocol (one run, not a grid search)

| Field | Choice |
|---|---|
| Source | existing **OVRSIS95K** versioned split, 35 source classes; reproducible random 10,000-image subset of source train |
| Source validation | its frozen source dev; do not use source dev labels for training or crop-benefit targets |
| External development set | existing DLRSD 200-image dev with its own 17-class vocabulary; not an independent held-out test |
| Host | same complete pretrained CAT-Seg ViT-B/16, **fully frozen**, same checkpoint and config SHA on both datasets |
| Selector | `full_rank` (Spatial CNN + Top-4 text cross-attention + action attention + BCE gain ranking) |
| Source cache | 1 augmentation per original image; 8 fixed real crops + first pass, no raw-cost hook |
| Selector training | batch **8**, AdamW, LR **1e-4**, 4,000 updates, seed **42**; fixed final checkpoint |
| Action and fusion | 50% and 75% corners, one extra real RGB crop, unchanged Sigmoid averaging |
| Default inference | forced one crop; No-op kept disabled for comparability |

**Why a subset?** Ten thousand real source images incur up to 90,000 CAT-Seg forward calls for the eight-action benefit cache. The `--sample-count` flag is a *formal deterministic random subset*, in contrast to `--limit` which is **smoke-only**. Sampling uses source-train membership and a fixed seed, never development GT. It is not claimed to be class-balanced or geographically independent; inspect the existing source split's provenance and class coverage before the run.

Use a V1-compatible manifest for each dataset: version 1; ignore label 255; train/val rows, `train_classes`, `test_classes`, and `seen_class_ids`; source-train masks in contiguous source IDs. OVRSIS95K must provide such a manifest; **the code does not assume that a different dataset's directory convention is already compatible**. V1 `Manifest` is the authority. For DLRSD, preserve the existing 14/17 mapping. Do not put dataset, cache, or checkpoints into Git.

## Commands (from repository root)

Set the real paths on the GPU machine:

```bash
export OVRS_MANIFEST=/path/to/ovrsis95k/manifest.json
export DLRSD_MANIFEST=/path/to/dlrsd/manifest.json
export CATSEG_CONFIG=/path/to/matching_vitb16.yaml
export CATSEG_WEIGHTS=/path/to/complete_pretrained_catseg.pth
export V2_ROOT=/path/outside/git/route_b_v2_ovrs10k_s42
```

Only add `--legacy-weights` when the original full checkpoint has no embedded config metadata (the V1 host still performs strict tensor loading).

1. **CPU tests, then 1–2 image CUDA smoke** in a separate throwaway cache using `--limit 2`; this smoke cache is intentionally refused for formal training without `--allow-smoke`.

```bash
python -m pytest -q research/ov_routes/v2/test_v2.py
python -m research.ov_routes.v2.run_v2 precompute \
  --manifest "$OVRS_MANIFEST" --config "$CATSEG_CONFIG" --weights "$CATSEG_WEIGHTS" \
  --device cuda --cache-dir "${V2_ROOT}_smoke" --limit 2 \
  --num-augmentations 1 --seed 42
```

2. **Generate one formal 10k source-only cache**. It records the exact sampled source indices, file-content hashes for those samples, the host/source-code fingerprints and the parity check result. Give the full cache an entirely new directory; resume of a partly generated cache is not supported by this prototype.

```bash
python -m research.ov_routes.v2.run_v2 precompute \
  --manifest "$OVRS_MANIFEST" --config "$CATSEG_CONFIG" --weights "$CATSEG_WEIGHTS" \
  --device cuda --cache-dir "${V2_ROOT}_cache" \
  --sample-count 10000 --num-augmentations 1 --seed 42
```

3. **Inspect labels once and train exactly one selector**.

```bash
python -m research.ov_routes.v2.run_v2 analyze-cache \
  --cache-dir "${V2_ROOT}_cache" --output "${V2_ROOT}_analysis"
python -m research.ov_routes.v2.run_v2 train \
  --cache-dir "${V2_ROOT}_cache" --output "${V2_ROOT}_train" \
  --ablation full_rank --device cuda --seed 42 \
  --steps 4000 --batch-size 8 --lr 0.0001
```

4. **Evaluate** the fixed final checkpoint on the source dev and external DLRSD dev; run the Zero baseline on both datasets with the identical frozen host. To use a different vocabulary, the code checks the identical model/config/source-code identity and compatible text-embedding width, **not** a falsely identical dataset manifest. GT is read for metrics only, never selector choice.

```bash
for dataset in ovrs dlrsd; do
  if [ "$dataset" = ovrs ]; then manifest="$OVRS_MANIFEST"; else manifest="$DLRSD_MANIFEST"; fi
  python -m research.ov_routes.v2.run_v2 evaluate \
    --manifest "$manifest" --config "$CATSEG_CONFIG" --weights "$CATSEG_WEIGHTS" \
    --device cuda --policy zero --output "${V2_ROOT}_${dataset}_zero"
  python -m research.ov_routes.v2.run_v2 evaluate \
    --manifest "$manifest" --config "$CATSEG_CONFIG" --weights "$CATSEG_WEIGHTS" \
    --device cuda --policy learned --selector-checkpoint "${V2_ROOT}_train/selector_final.pt" \
    --output "${V2_ROOT}_${dataset}_learned"
done
```

Each output directory must not already exist; the runner intentionally refuses overwrites. `analyze-cache` reports BCE/IoU direction mismatch and candidate-level opportunity on **source train only**. `train` emits internal holdout diagnostics. `evaluate` records total/seen/unseen/hIoU, selected actions, times and encoder call counts. Be explicit: when OVRSIS95K source classes overlap DLRSD held-out names, DLRSD is **cross-dataset** but not strictly pretrained-unseen.

## Audit boundaries

- CPU tests use toy hosts; they do not prove CUDA checkpoint compatibility. Real Host native parity is checked at precompute start.
- Exact model identity (weights/config/CAT-Seg code) is enforced for cross-dataset selector reuse; train/development manifests intentionally differ.
- A new Host checkpoint requires a *new* source cache, selector training and Zero baseline. Do **not** compare a new host's results against the old 29.629% Zero as though they were matched.
- An improvement in BCE benefit prediction is not automatically an improvement in aggregate mIoU. Zero vs learned external dev scores decide this pilot. No sealed test access, no multi-seed search, no sweeping of H1–H4 for this run.
- This version stores its own runtime outputs **outside the repository**; `.gitignore` already excludes `*.pt` and other artifacts.
