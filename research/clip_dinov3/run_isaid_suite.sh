#!/usr/bin/env bash
# Matched C-series exploration on full or held-class iSAID.
set -euo pipefail
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo 'Usage: bash run_isaid_suite.sh preflight|main|external MANIFEST CLIP DINO OUT [BATCH=8]' >&2
  exit 2
fi
phase=$1
manifest=$(realpath "$2"); clip=$(realpath "$3"); dino=$(realpath "$4")
out=$(realpath -m "$5"); batch=${6:-8}
epochs=${CSERIES_EPOCHS:-5}
[[ "$epochs" == 5 || "$epochs" == 15 || "$epochs" == 40 ]] ||
  { echo 'CSERIES_EPOCHS must be 5, 15 or 40' >&2; exit 2; }
printf -v final_epoch '%03d' "$epochs"
seed=${CSERIES_SEED:-42}
[[ "$seed" =~ ^(0|[1-9][0-9]{0,9})$ ]] || { echo 'Invalid CSERIES_SEED' >&2; exit 2; }
held=${CSERIES_HELD_SPLIT:-}
unattended=${CSERIES_UNATTENDED:-0}
[[ "$unattended" == 0 || "$unattended" == 1 ]] || { echo 'Invalid CSERIES_UNATTENDED' >&2; exit 2; }
lane_a=(C0 D0 C2 C3 C5 E01)
lane_b=(B0 C1 C4 C6 E02)
allowed=(C0 B0 D0 C1 C2 C3 C4 C5 C6 E01 E02)
defaults=C0,B0,D0,C1,C2,C3,C4,C5,C6
if [[ -n "$held" ]]; then
  [[ "$held" == H1 || "$held" == H2 ]] || { echo 'CSERIES_HELD_SPLIT must be H1 or H2' >&2; exit 2; }
  allowed=(C0 B0 D0 C1 C2 E01 E02)
  defaults=C0,B0
fi
selection=${CSERIES_VARIANTS:-$defaults}
[[ "$selection" =~ ^[A-Z][0-9]{1,2}(,[A-Z][0-9]{1,2})*$ ]] || { echo 'Invalid CSERIES_VARIANTS' >&2; exit 2; }
IFS=, read -r -a variants <<< "$selection"
selected=,
for v in "${variants[@]}"; do
  [[ " ${allowed[*]} " == *" $v "* && "$selected" != *",$v,"* ]] ||
    { echo "Unsupported or repeated variant: $v" >&2; exit 2; }
  selected+="$v,"
done
gpu0=; gpu1=
if [[ -n "${CSERIES_GPU_IDS:-}" ]]; then
  IFS=, read -r gpu0 gpu1 <<< "$CSERIES_GPU_IDS"
  [[ -n "$gpu0" && -n "$gpu1" && "$gpu0" != "$gpu1" && "$gpu1" != *,* ]] ||
    { echo 'CSERIES_GPU_IDS must contain two distinct comma-separated GPU IDs' >&2; exit 2; }
fi
# Opt in before main training if exact continuation of completed arms is required.
keep_final_state=${KEEP_FINAL_STATE:-0}
[[ "$keep_final_state" == 0 || "$keep_final_state" == 1 ]] || { echo 'KEEP_FINAL_STATE must be 0 or 1' >&2; exit 2; }
train_policy=(--lr 1e-3 --weight-decay .01 --lr-schedule cosine --train-warmup-steps 300 --min-lr-ratio .1 --grad-clip 1 --save-every 500)
[[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid batch'; exit 2; }
[[ "$phase" == preflight || "$phase" == main || "$phase" == external ]] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=${PYTHON:-python}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$out"
exec 9>"$out/.suite.lock"
flock -n 9 || { echo "Another suite process owns $out" >&2; exit 2; }
snapshot_schedule=$("$PYTHON" -c 'import sys; from isaid_protocol import development_epochs; print(*(f"{e:03d}" for e in development_epochs(int(sys.argv[1]))))' "$epochs")
read -r -a dev_epochs <<< "$snapshot_schedule"
# Check the original published vocabulary/split before launching any job.
"$PYTHON" - "$manifest" "$clip" "$dino" <<'PY'
import json, sys
from pathlib import Path
from prepare_isaid import CLASSES, SPLIT_SIZES
p=Path(sys.argv[1]); m=json.loads(p.read_text(encoding='utf-8'))
if m['ignore_label'] != 255:
    raise ValueError('iSAID ignore label must be 255')
for field in ('train_classes','test_classes'):
    if json.loads((p.parent/m[field]).read_text(encoding='utf-8')) != CLASSES:
        raise ValueError(f'Unexpected iSAID vocabulary: {field}')
ids={}
for split,n in SPLIT_SIZES.items():
    rows=m[split]; ids[split]={r['sample_id'] for r in rows}
    if len(rows) != n or len(ids[split]) != n:
        raise ValueError(f'{split}: expected {n} unique sample IDs, found {len(rows)} rows and {len(ids[split])} IDs')
    for r in rows:
        for key in ('file_name','sem_seg_file_name'):
            file = p.parent/r[key]
            if file.suffix != '.png' or file.stem != r['sample_id']:
                raise ValueError(f'{split}: {key} does not match PNG sample {r["sample_id"]}: {file}')
            if not file.is_file():
                raise FileNotFoundError(file)
        if split=='train' and r.get('pixel_status_file_name'):
            raise ValueError('Use the original iSAID manifest, not an extra supervision manifest')
if ids['train'] & ids['val']:
    raise ValueError('Train/val sample ID overlap')
for f in sys.argv[2:]:
    if not Path(f).is_file(): raise FileNotFoundError(f)
print('Manifest and weight paths checked; runner verifies weight hashes and loaded masks.')
PY
# Bind the preflight to the selected models, data and optimizer horizon.
source=$manifest
if [[ -n "$held" ]]; then
  manifest="$out/held_data/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    [[ "$phase" == preflight ]] || { echo 'Generate held-class data with preflight first' >&2; exit 2; }
    "$PYTHON" -c 'from isaid_protocol import prepare_isaid_held; import sys; prepare_isaid_held(*sys.argv[1:])' "$source" "$out/held_data" "$held"
  fi
  held_key=$("$PYTHON" -c 'from isaid_protocol import held_identity; import sys; print(held_identity(*sys.argv[1:]))' "$source" "$manifest" "$held")
else
  held_key=
fi
key=$(printf '%s\n' "$(git rev-parse HEAD)" "$(sha256sum "$source")" "$clip" "$dino" "$batch" "$epochs" "${train_policy[*]}")
if [[ "$selection" != C0,B0,D0,C1,C2,C3,C4,C5,C6 || "$seed" != 42 || -n "$held" ]]; then
  key=$(printf '%s\n' "$key" "$selection" "$seed" "$held" "$held_key")
fi
if [[ -e "$out/protocol.txt" ]]; then
  [[ "$(cat "$out/protocol.txt")" == "$key" ]] || { echo 'Protocol changed; choose a new OUT directory' >&2; exit 2; }
else
  [[ "$phase" == preflight ]] || { echo 'Run preflight first' >&2; exit 2; }
  printf '%s\n' "$key" > "$out/protocol.txt"
fi
common=(--manifest "$manifest" --clip-weights "$clip" --size 384 --device cuda --seed "$seed")
tracked=0
loss_flags=()
if [[ "$epochs" == 40 || -n "$held" ]]; then
  tracked=1
  loss_flags=(--eval-loss)
fi
check_report() {
  "$PYTHON" - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
from pathlib import Path
path, variant, kind, expected = sys.argv[1:]
r = json.loads(Path(path).read_text(encoding='utf-8'))
if r['config']['variant'] != variant:
    raise ValueError(f'Unexpected variant: {path}')
if kind == 'train':
    if r['completed_epochs'] != int(expected) or r['steps'] <= 0:
        raise ValueError(f'Training is incomplete for epoch {expected}: {path}')
else:
    if (r['checkpoint_epoch'] != int(expected) or r['checkpoint_kind'] != 'evaluation_only'
            or r['checkpoint_steps'] <= 0 or r['images'] <= 0 or r['miou_percent'] is None):
        raise ValueError(f'Evaluation is incomplete for epoch {expected}: {path}')
PY
}
run_job() {
  local dest=$1 variant=$2 expected=$3 kind=$4
  shift 3
  local -a recovery=()
  if [[ -f "$dest/report.json" ]] && check_report "$dest/report.json" "$variant" "$kind" "$expected" 2>/dev/null; then
    echo "Completed: $dest"
    return
  fi
  if [[ -e "$dest/state.pt" ]]; then
    if [[ "$unattended" == 1 && "$kind" == train ]]; then
      recovery=(--checkpoint "$dest/state.pt")
      echo "Resuming $dest from rolling state"
    else
      echo "Interrupted checkpoint at $dest; resume explicitly with run_combinations.py before continuing." >&2; exit 2
    fi
  fi
  mkdir -p "$dest"
  "$PYTHON" -u run_combinations.py "$@" "${recovery[@]}" --output "$dest" 2>&1 | tee -a "$dest/console.log"
  check_report "$dest/report.json" "$variant" "$kind" "$expected"
}
if [[ "$phase" != preflight ]]; then
  [[ -f "$out/preflight.ok" ]] && [[ "$(cat "$out/preflight.ok")" == "$key" ]] || { echo 'Matching preflight has not completed' >&2; exit 2; }
fi
if [[ "$phase" == external ]]; then
  : "${EXTERNAL_MANIFEST:?Set EXTERNAL_MANIFEST to an audited external val manifest}"
  external=$(realpath "$EXTERNAL_MANIFEST")
  [[ "$external" != "$manifest" && "$external" != "$source" ]] || { echo 'External manifest must differ from training manifest'; exit 2; }
  tag=$(sha256sum "$external" | cut -c1-16)
fi
run_variant() {
  local v=$1 dest e
  local -a weights=() extras=()
  [[ "$v" == C0 ]] || weights=(--dino-weights "$dino")
  [[ "$v" != C5 && "$v" != C6 ]] || extras=(--no-pseudo)
  [[ "$v" != C6 ]] || extras+=(--no-route-train)
  if [[ "$phase" == preflight ]]; then
    dest="$out/preflight/$v"
    run_job "$dest/train" "$v" 1 train "${common[@]}" "${weights[@]}" --variant "$v" --batch-size "$batch" --epochs 1 --lr 1e-3 --weight-decay .01 --lr-schedule constant --train-warmup-steps 0 --grad-clip 1 --workers 0 --limit "$((batch*3))" --audit "${extras[@]}"
    run_job "$dest/eval" "$v" 1 eval "${common[@]}" "${weights[@]}" --batch-size 1 --workers 0 --limit 8 --warmup 2 --checkpoint "$dest/train/final.pt"
    # Smoke weights have no research value after successful checks; keep reports.
    rm -f -- "$dest/train/state.pt" "$dest/train/final.pt"
  elif [[ "$phase" == main ]]; then
    dest="$out/main/$v"
    run_job "$dest/train" "$v" "$epochs" train "${common[@]}" "${weights[@]}" --variant "$v" --batch-size "$batch" --epochs "$epochs" "${train_policy[@]}" --workers 4 --audit "${extras[@]}"
    [[ -s "$dest/train/final.pt" ]] || { echo "Missing final export: $v" >&2; exit 2; }
    # Same deterministic 512-image source validation subset at every checkpoint.
    # Delete each intermediate export only after its completed report is checked.
    for e in "${dev_epochs[@]}"; do
      run_job "$dest/dev_epoch_$e" "$v" "$((10#$e))" eval "${common[@]}" "${weights[@]}" "${loss_flags[@]}" --batch-size 1 --workers 2 --warmup 0 --eval-samples 512 --eval-sample-seed 42 --checkpoint "$dest/train/epoch_$e.pt"
      if (( ! tracked )); then rm -f -- "$dest/train/epoch_$e.pt"; fi
    done
    run_job "$dest/dev_epoch_$final_epoch" "$v" "$epochs" eval "${common[@]}" "${weights[@]}" "${loss_flags[@]}" --batch-size 1 --workers 2 --warmup 0 --eval-samples 512 --eval-sample-seed 42 --checkpoint "$dest/train/final.pt"
    run_job "$dest/eval" "$v" "$epochs" eval "${common[@]}" "${weights[@]}" "${loss_flags[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$dest/train/final.pt"
    if [[ "$v" == C6 ]]; then
      run_job "$dest/keep_all" "$v" "$epochs" eval "${common[@]}" "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$dest/train/final.pt" --force-keep-all
    fi
    if (( tracked )); then
      "$PYTHON" -c 'from isaid_protocol import finish_tracked_run; import sys; finish_tracked_run(*sys.argv[1:])' "$dest" "$epochs" "$held"
    fi
    # All planned local evaluations succeeded: retain final model for external
    # eval, drop optimizer/RNG state by default. No deletion of shared weights.
    if [[ "$keep_final_state" == 0 ]]; then
      rm -f -- "$dest/train/state.pt"
    fi
  else
    dest="$out/external/$tag/$v"
    check_report "$out/main/$v/train/report.json" "$v" train "$epochs"
    if [[ "$epochs" == 40 ]]; then
      for e in 015 030; do
        run_job "$dest/epoch_$e" "$v" "$((10#$e))" eval --manifest "$external" --clip-weights "$clip" --size 384 --device cuda --seed "$seed" "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$out/main/$v/train/epoch_$e.pt"
      done
    fi
    run_job "$dest/eval" "$v" "$epochs" eval --manifest "$external" --clip-weights "$clip" --size 384 --device cuda --seed "$seed" "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$out/main/$v/train/final.pt"
  fi
}
run_lane() {
  local gpu=$1 v job failed=0
  shift
  if [[ -n "$gpu" ]]; then export CUDA_VISIBLE_DEVICES="$gpu"; fi
  printf 'phase=%s gpu=%s variants=%s\n' "$phase" "${CUDA_VISIBLE_DEVICES:-default}" "$*"
  for v in "$@"; do
    [[ "$selected" == *",$v,"* ]] || continue
    if [[ "$unattended" == 1 ]]; then
      run_variant "$v" &
      job=$!
      if ! wait "$job"; then
        echo "Failed: $phase/$v (other selected variants continue)" >&2
        failed=1
      fi
    else
      run_variant "$v"
    fi
  done
  return "$failed"
}
if [[ -n "$gpu0" ]]; then
  # Balance estimated durations while retaining the relative order on each GPU.
  run_lane "$gpu0" "${lane_a[@]}" &
  lane0=$!
  run_lane "$gpu1" "${lane_b[@]}" &
  lane1=$!
  failed=0
  wait "$lane0" || failed=1
  wait "$lane1" || failed=1
  if (( failed )); then exit 1; fi
else
  run_lane "" "${variants[@]}"
fi
if [[ "$phase" == preflight ]]; then
  printf '%s\n' "$key" > "$out/preflight.ok"
else
  root="$out/main"; [[ "$phase" != external ]] || root="$out/external/$tag"
  "$PYTHON" - "$root" "$selection" "$phase" "$epochs" "${dev_epochs[@]}" "$final_epoch" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
checks=('eval', 'keep_all', *(f'dev_epoch_{epoch}' for epoch in sys.argv[5:]))
if sys.argv[3:5] == ['external', '40']:
    checks=('epoch_015', 'epoch_030', 'eval')
for v in sys.argv[2].split(','):
    for sub in checks:
        p=root/v/sub/'report.json'
        if p.is_file():
            r=json.loads(p.read_text(encoding='utf-8'))
            row={'variant':v, 'mode':sub, **{k:r.get(k) for k in ('miou_percent','seen_miou','unseen_miou','harmonic_iou','trainable_parameters','images','ms_per_image','peak_cuda_mib','mean_retained_classes','class_presence_recall','pixel_recall','sample_ids_sha256')}}
            if 'mean_pixel_cross_entropy' in r:
                row['mean_pixel_cross_entropy'] = r['mean_pixel_cross_entropy']
            rows.append(row)
with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
    fields=list(dict.fromkeys(k for row in rows for k in row))
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print(root/'summary.csv')
PY
fi
