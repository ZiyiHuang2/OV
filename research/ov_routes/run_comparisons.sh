#!/usr/bin/env bash
set -euo pipefail
: "${OV_MANIFEST:?Set existing version-1 train/val manifest}"
: "${OV_CONFIG:?Set matching full pretrained CAT-Seg YAML}"
: "${OV_WEIGHTS:?Set complete pretrained segmentation checkpoint}"
: "${OV_OUTPUT:?Set NEW output root}"
: "${A_STEPS:?Choose explicit total A updates}"
: "${B_STEPS:?Choose explicit total B updates}"
common=(--manifest "$OV_MANIFEST" --config "$OV_CONFIG" --weights "$OV_WEIGHTS" --seed 42)
if [[ "${OV_LEGACY_WEIGHTS:-0}" == 1 ]]; then common+=(--legacy-weights); fi
if [[ -e "$OV_OUTPUT" ]]; then echo 'Output root already exists; choose a fresh directory' >&2; exit 1; fi
mkdir -p "$OV_OUTPUT"
python -m research.ov_routes.run evaluate "${common[@]}" --budget 0 --output "$OV_OUTPUT/b_zero"
for policy in fixed random; do
  python -m research.ov_routes.run evaluate "${common[@]}" --policy "$policy" --budget 1 --output "$OV_OUTPUT/b_$policy"
done
for mode in erm joint meta; do
  python -m research.ov_routes.run train-a "${common[@]}" --mode "$mode" --steps "$A_STEPS" \
    --budget 0 --output "$OV_OUTPUT/a_$mode"
done
python -m research.ov_routes.run train-b "${common[@]}" --steps "$B_STEPS" --lr .0002 \
  --budget 1 --output "$OV_OUTPUT/b_learned"
