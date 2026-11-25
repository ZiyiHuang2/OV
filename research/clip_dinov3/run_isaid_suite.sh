#!/usr/bin/env bash
# Matched five-epoch C-series exploration. No automatic batch-size change.
set -euo pipefail
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo 'Usage: bash run_isaid_suite.sh preflight|main|external MANIFEST CLIP DINO OUT [BATCH=8]' >&2
  exit 2
fi
phase=$1
manifest=$(realpath "$2"); clip=$(realpath "$3"); dino=$(realpath "$4")
out=$(realpath -m "$5"); batch=${6:-8}
epochs=5
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
# All nine configurations use one fixed protocol; bind the preflight to code/data/paths.
key=$(printf '%s\n' "$(git rev-parse HEAD)" "$(sha256sum "$manifest")" "$clip" "$dino" "$batch" "$epochs" "${train_policy[*]}")
if [[ -e "$out/protocol.txt" ]]; then
  [[ "$(cat "$out/protocol.txt")" == "$key" ]] || { echo 'Protocol changed; choose a new OUT directory' >&2; exit 2; }
else
  [[ "$phase" == preflight ]] || { echo 'Run preflight first' >&2; exit 2; }
  printf '%s\n' "$key" > "$out/protocol.txt"
fi
variants=(C0 B0 D0 C1 C2 C3 C4 C5 C6)
common=(--manifest "$manifest" --clip-weights "$clip" --size 384 --device cuda --seed 42)
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
  if [[ -f "$dest/report.json" ]] && check_report "$dest/report.json" "$variant" "$kind" "$expected" 2>/dev/null; then
    echo "Completed: $dest"
    return
  fi
  if [[ -e "$dest/state.pt" ]]; then
    echo "Interrupted checkpoint at $dest; resume explicitly with run_combinations.py before continuing." >&2; exit 2
  fi
  mkdir -p "$dest"
  "$PYTHON" -u run_combinations.py "$@" --output "$dest" 2>&1 | tee "$dest/console.log"
  check_report "$dest/report.json" "$variant" "$kind" "$expected"
}
if [[ "$phase" != preflight ]]; then
  [[ -f "$out/preflight.ok" ]] && [[ "$(cat "$out/preflight.ok")" == "$key" ]] || { echo 'Matching preflight has not completed' >&2; exit 2; }
fi
for v in "${variants[@]}"; do
  weights=(); [[ "$v" == C0 ]] || weights=(--dino-weights "$dino")
  extras=(); [[ "$v" != C5 && "$v" != C6 ]] || extras=(--no-pseudo)
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
    # Same deterministic 512-image source validation subset at epochs 1, 3, 5.
    # Delete each intermediate export only after its completed report is checked.
    for e in 001 003; do
      run_job "$dest/dev_epoch_$e" "$v" "$((10#$e))" eval "${common[@]}" "${weights[@]}" --batch-size 1 --workers 2 --warmup 0 --eval-samples 512 --eval-sample-seed 42 --checkpoint "$dest/train/epoch_$e.pt"
      rm -f -- "$dest/train/epoch_$e.pt"
    done
    run_job "$dest/dev_epoch_005" "$v" "$epochs" eval "${common[@]}" "${weights[@]}" --batch-size 1 --workers 2 --warmup 0 --eval-samples 512 --eval-sample-seed 42 --checkpoint "$dest/train/final.pt"
    run_job "$dest/eval" "$v" "$epochs" eval "${common[@]}" "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$dest/train/final.pt"
    if [[ "$v" == C6 ]]; then
      run_job "$dest/keep_all" "$v" "$epochs" eval "${common[@]}" "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$dest/train/final.pt" --force-keep-all
    fi
    # All planned local evaluations succeeded: retain final model for external
    # eval, drop optimizer/RNG state by default. No deletion of shared weights.
    if [[ "$keep_final_state" == 0 ]]; then
      rm -f -- "$dest/train/state.pt"
    fi
  else
    : "${EXTERNAL_MANIFEST:?Set EXTERNAL_MANIFEST to an audited external val manifest}"
    external=$(realpath "$EXTERNAL_MANIFEST")
    [[ "$external" != "$manifest" ]] || { echo 'External manifest must differ from training manifest'; exit 2; }
    tag=$(sha256sum "$external" | cut -c1-16)
    dest="$out/external/$tag/$v"
    check_report "$out/main/$v/train/report.json" "$v" train "$epochs"
    run_job "$dest/eval" "$v" "$epochs" eval --manifest "$external" --clip-weights "$clip" --size 384 --device cuda --seed 42 "${weights[@]}" --batch-size 1 --workers 2 --warmup 30 --checkpoint "$out/main/$v/train/final.pt"
  fi
done
if [[ "$phase" == preflight ]]; then
  printf '%s\n' "$key" > "$out/preflight.ok"
else
  root="$out/main"; [[ "$phase" != external ]] || root="$out/external/$tag"
  "$PYTHON" - "$root" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for v in ('C0','B0','D0','C1','C2','C3','C4','C5','C6'):
    for sub in ('eval','keep_all','dev_epoch_001','dev_epoch_003','dev_epoch_005'):
        p=root/v/sub/'report.json'
        if p.is_file():
            r=json.loads(p.read_text(encoding='utf-8'))
            row={'variant':v, 'mode':sub, **{k:r.get(k) for k in ('miou_percent','seen_miou','unseen_miou','harmonic_iou','trainable_parameters','images','ms_per_image','peak_cuda_mib','mean_retained_classes','class_presence_recall','pixel_recall','sample_ids_sha256')}}
            rows.append(row)
with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(root/'summary.csv')
PY
fi
