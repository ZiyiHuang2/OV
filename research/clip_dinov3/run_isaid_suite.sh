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
lane_a=(C0 D0 C2 C3 C5 E01 D01 F01 F08)
lane_b=(B0 C1 C4 C6 E02 D02 F02 F09)
allowed=("${lane_a[@]}" "${lane_b[@]}")
defaults=C0,B0,D0,C1,C2,C3,C4,C5,C6
if [[ -n "$held" ]]; then
  [[ "$held" == H1 || "$held" == H2 ]] || { echo 'CSERIES_HELD_SPLIT must be H1 or H2' >&2; exit 2; }
  allowed=(C0 B0 D0 C1 C2 D01 D02 E01 E02 F01 F02 F08 F09)
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
size=${CSERIES_SIZE:-384}
[[ "$size" =~ ^[1-9][0-9]{1,3}$ ]] && (( size % 16 == 0 )) ||
  { echo 'CSERIES_SIZE must be a positive multiple of 16' >&2; exit 2; }
c2_control=${CSERIES_C2_CONTROL:-}
c2_flags=()
case "$c2_control" in
  '') ;;
  mean|learned|separate) c2_flags=(--fusion "$c2_control") ;;
  noalign) c2_flags=(--alignment-weight 0) ;;
  classpool) c2_flags=(--region-mode class) ;;
  *) echo 'Invalid CSERIES_C2_CONTROL' >&2; exit 2 ;;
esac
[[ -z "$c2_control" || "$selection" == C2 ]] ||
  { echo 'CSERIES_C2_CONTROL requires CSERIES_VARIANTS=C2' >&2; exit 2; }
gpu0=; gpu1=
if [[ -n "${CSERIES_GPU_IDS:-}" ]]; then
  IFS=, read -r gpu0 gpu1 <<< "$CSERIES_GPU_IDS"
  [[ -n "$gpu0" && -n "$gpu1" && "$gpu0" != "$gpu1" && "$gpu1" != *,* ]] ||
    { echo 'CSERIES_GPU_IDS must contain two distinct comma-separated GPU IDs' >&2; exit 2; }
fi
# Preserve exact recovery for the resolved horizon unless cleanup is explicitly requested.
keep_final_state=${KEEP_FINAL_STATE:-1}
[[ "$keep_final_state" == 0 || "$keep_final_state" == 1 ]] || { echo 'KEEP_FINAL_STATE must be 0 or 1' >&2; exit 2; }
track_every_two=${CSERIES_TRACK_DEVELOPMENT:-0}
[[ "$track_every_two" == 0 || "$track_every_two" == 1 ]] ||
  { echo 'CSERIES_TRACK_DEVELOPMENT must be 0 or 1' >&2; exit 2; }
train_policy=(--lr 1e-3 --weight-decay .01 --lr-schedule cosine --train-warmup-steps 300 --min-lr-ratio .1 --grad-clip 1 --save-every 500)
if (( track_every_two )); then train_policy+=(--track-development); fi
[[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid batch'; exit 2; }
[[ "$phase" == preflight || "$phase" == main || "$phase" == external ]] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=${PYTHON:-python}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$out"
exec 9>"$out/.suite.lock"
flock -n 9 || { echo "Another suite process owns $out" >&2; exit 2; }
snapshot_schedule=$("$PYTHON" -c 'import sys; from isaid_protocol import development_epochs; print(*(f"{e:03d}" for e in development_epochs(int(sys.argv[1]), sys.argv[2]=="1")))' "$epochs" "$track_every_two")
read -r -a dev_epochs <<< "$snapshot_schedule"
# The queue shares one full decoded data audit across jobs. A standalone
# invocation keeps its audit alongside its preflight and protocol.
[[ -f "$clip" && -f "$dino" ]] || { echo 'Missing encoder weights' >&2; exit 2; }
audit_report=${CSERIES_DATA_AUDIT:-"$out/isaid_data_audit.json"}
source_data_hash=$("$PYTHON" -c 'from isaid_protocol import audit_isaid; import sys; print(audit_isaid(*sys.argv[1:])["data_sha256"])' "$manifest" "$audit_report")
# Bind the preflight to the selected models, data and optimizer horizon.
source=$manifest
if [[ -n "$held" ]]; then
  manifest="$out/held_data/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    [[ "$phase" == preflight ]] || { echo 'Generate held-class data with preflight first' >&2; exit 2; }
    "$PYTHON" -c 'from isaid_protocol import prepare_isaid_held; import sys; prepare_isaid_held(*sys.argv[1:])' "$source" "$out/held_data" "$held" "$source_data_hash"
  fi
  held_key=$("$PYTHON" -c 'from isaid_protocol import held_identity; import sys; print(held_identity(*sys.argv[1:]))' "$source" "$manifest" "$held" "$source_data_hash")
else
  held_key=
fi
key=$(printf '%s\n' "$(git rev-parse HEAD)" "$(sha256sum "$source")" "$source_data_hash" "$clip" "$dino" "$batch" "$epochs" "${train_policy[*]}")
if [[ "$selection" != C0,B0,D0,C1,C2,C3,C4,C5,C6 || "$seed" != 42 || -n "$held" ]]; then
  key=$(printf '%s\n' "$key" "$selection" "$seed" "$held" "$held_key")
fi
if [[ "$size" != 384 || -n "$c2_control" ]]; then
  key=$(printf '%s\n' "$key" "$size" "$c2_control")
fi
if [[ -e "$out/protocol.txt" ]]; then
  [[ "$(cat "$out/protocol.txt")" == "$key" ]] || { echo 'Protocol changed; choose a new OUT directory' >&2; exit 2; }
else
  [[ "$phase" == preflight ]] || { echo 'Run preflight first' >&2; exit 2; }
  printf '%s\n' "$key" > "$out/protocol.txt"
fi
common=(--manifest "$manifest" --data-sha256 "$source_data_hash" --clip-weights "$clip" --size "$size" --device cuda --seed "$seed")
tracked=0
loss_flags=()
if [[ "$epochs" == 40 || -n "$held" || "$track_every_two" == 1 ]]; then
  tracked=1
  loss_flags=(--eval-loss)
fi
check_report() {
  "$PYTHON" - "$1" "$2" "$3" "$4" "${5:-}" <<'PY'
import json, sys
from pathlib import Path
from isaid_protocol import file_sha256
path, variant, kind, expected, checkpoint = sys.argv[1:]
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
    if checkpoint and Path(checkpoint).is_file() and r.get('checkpoint_sha256') != file_sha256(Path(checkpoint)):
        raise ValueError(f'Evaluation checkpoint content changed: {checkpoint}')
PY
}
run_job() {
  local dest=$1 variant=$2 expected=$3 kind=$4
  shift 3
  local -a recovery=()
  local checkpoint= value previous=
  if [[ "$kind" == eval ]]; then
    for value in "$@"; do
      if [[ "$previous" == --checkpoint ]]; then checkpoint=$value; break; fi
      previous=$value
    done
  fi
  if [[ -f "$dest/report.json" ]] && check_report "$dest/report.json" "$variant" "$kind" "$expected" "$checkpoint" 2>/dev/null; then
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
  check_report "$dest/report.json" "$variant" "$kind" "$expected" "$checkpoint"
}
if [[ "$phase" != preflight ]]; then
  [[ -f "$out/preflight.ok" ]] && [[ "$(cat "$out/preflight.ok")" == "$key" ]] || { echo 'Matching preflight has not completed' >&2; exit 2; }
fi
if [[ "$phase" == external ]]; then
  : "${EXTERNAL_MANIFEST:?Set EXTERNAL_MANIFEST to an audited external val manifest}"
  external=$(realpath "$EXTERNAL_MANIFEST")
  "$PYTHON" -c 'from isaid_protocol import check_external_source; import sys; check_external_source(*sys.argv[1:])' "$source" "$external"
  tag=$(sha256sum "$external" | cut -c1-16)
fi
run_variant() {
  local v=$1 dest e
  local -a weights=() extras=()
  [[ "$v" == C0 || "$v" == D02 ]] || weights=(--dino-weights "$dino")
  [[ "$v" != C2 ]] || extras+=("${c2_flags[@]}")
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
      "$PYTHON" -c 'from isaid_protocol import finish_tracked_run; import sys; finish_tracked_run(*sys.argv[1:4], sys.argv[4]=="1")' "$dest" "$epochs" "$held" "$track_every_two"
    fi
    # All planned local evaluations succeeded: retain final model for external
    # eval, drop optimizer/RNG state only if requested. No shared weights are deleted.
    if [[ "$keep_final_state" == 0 ]]; then
      rm -f -- "$dest/train/state.pt"
    fi
  else
    dest="$out/external/$tag/$v"
    check_report "$out/main/$v/train/report.json" "$v" train "$epochs"
    if [[ "$epochs" == 40 ]]; then
      for e in 015 030; do
        run_job "$dest/epoch_$e" "$v" "$((10#$e))" eval --manifest "$external" --clip-weights "$clip" --size "$size" --device cuda --seed "$seed" "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$out/main/$v/train/epoch_$e.pt"
      done
    fi
    run_job "$dest/eval" "$v" "$epochs" eval --manifest "$external" --clip-weights "$clip" --size "$size" --device cuda --seed "$seed" "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$out/main/$v/train/final.pt"
  fi
}
run_lane() {
  local gpu=$1 v job status failed=0
  shift
  if [[ -n "$gpu" ]]; then export CUDA_VISIBLE_DEVICES="$gpu"; fi
  printf 'phase=%s gpu=%s variants=%s\n' "$phase" "${CUDA_VISIBLE_DEVICES:-default}" "$*"
  for v in "$@"; do
    [[ "$selected" == *",$v,"* ]] || continue
    if [[ "$unattended" == 1 ]]; then
      run_variant "$v" &
      job=$!
      if wait "$job"; then
        continue
      else
        status=$?
        echo "Failed: $phase/$v (exit $status, other selected variants continue)" >&2
        if (( ! failed )); then failed=$status; fi
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
