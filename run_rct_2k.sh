#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
base=/home/wangchengjun/huangziyi/open-ov/asset_store/checkpoints/cat_seg/model_base.pth
manifest=/home/wangchengjun/huangziyi/ov_competition_v1_20261030_audit/frozen_split_seed42_v2/manifest.json
python=/home/wangchengjun/miniconda3/envs/ovrs/bin/python
root=/tmp/ov-rct-validation-20261115
cd "$root/repo"
for arm in A C M V R; do
  if [[ "$arm" == A || "$arm" == C ]]; then
    method=(--competition-arm "$arm" --competition-base-weights "$base")
  else
    method=(--rival-arm "$arm" --rival-base-weights "$base")
  fi
  printf '%s START %s\n' "$(date -Is)" "$arm" | tee -a "$root/exploration_status.log"
  "$python" train_ovrs.py --dataset-manifest "$manifest" --experiment baseline \
    --weights "$base" "${method[@]}" --output "$root/explore_$arm" \
    --batch-size 4 --max-iter 2000 SOLVER.BASE_LR 0.0001 \
    SOLVER.AMP.ENABLED False SOLVER.CHECKPOINT_PERIOD 5000 TEST.EVAL_PERIOD 0 \
    DATALOADER.NUM_WORKERS 4 SEED 42 > "$root/explore_$arm.log" 2>&1
  printf '%s COMPLETE %s\n' "$(date -Is)" "$arm" | tee -a "$root/exploration_status.log"
done
