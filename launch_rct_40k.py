"""Run the frozen three-arm protocol and monitor shared GPU/disk usage."""
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

repo = Path(__file__).resolve().parent
root = repo.parent
manifest = '/home/wangchengjun/huangziyi/ov_competition_v1_20261030_audit/frozen_split_seed42_v2/manifest.json'
base = '/home/wangchengjun/huangziyi/open-ov/asset_store/checkpoints/cat_seg/model_base.pth'
python = '/home/wangchengjun/miniconda3/envs/ovrs/bin/python'
arms = {'M': '0', 'V': '1', 'R': '2'}
if any((root / arm).exists() for arm in arms):
    raise RuntimeError('Refusing to reuse an existing training output directory')
status_path = root / 'formal_status.json'
if status_path.exists():
    raise RuntimeError('Formal status already exists')
status = {'state': 'starting', 'arms': {}, 'seed': 42, 'max_iter': 40000,
          'evaluation_interval': 5000, 'checkpoint_interval': 5000,
          'selection': 'highest source-dev mIoU at 5000-step points, earliest on ties',
          'manifest': manifest, 'base_checkpoint': base,
          'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
          'note': 'Resume restores optimizer/scheduler; exact random/data-order continuity not established.'}
def save_status():
    temporary = status_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2))
    os.replace(temporary, status_path)
save_status()
procs = {}
files = []
try:
    for arm, gpu in arms.items():
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        command = [python, 'train_ovrs.py', '--dataset-manifest', manifest, '--experiment', 'baseline',
                   '--weights', base, '--rival-arm', arm, '--rival-base-weights', base,
                   '--output', str(root / arm), '--batch-size', '4', '--max-iter', '40000',
                   'SOLVER.BASE_LR', '0.0001', 'SOLVER.AMP.ENABLED', 'False',
                   'SOLVER.CHECKPOINT_PERIOD', '5000', 'TEST.EVAL_PERIOD', '5000',
                   'MODEL.OVRS.COMPACT_OUTPUT', 'True', 'DATALOADER.NUM_WORKERS', '4', 'SEED', '42']
        log = (root / f'train_{arm}.log').open('x')
        files.append(log)
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        procs[arm] = proc
        status['arms'][arm] = {'pid': proc.pid, 'gpu': int(gpu), 'command': command}
        save_status()
    status['state'] = 'running'; save_status()
    with (root / 'resource_monitor.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['utc', 'disk_free_gib', 'gpu0_free_mib', 'gpu1_free_mib', 'gpu2_free_mib',
                         'gpu0_util_pct', 'gpu1_util_pct', 'gpu2_util_pct', 'M_exit', 'V_exit', 'R_exit'])
        low_gpu_samples = 0
        while any(p.poll() is None for p in procs.values()):
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
            gpu = [[int(x.strip()) for x in row.split(',')] for row in raw.strip().splitlines()]
            free = shutil.disk_usage(root).free / 2**30
            codes = [procs[a].poll() for a in arms]
            writer.writerow([datetime.now(timezone.utc).isoformat(), round(free, 2),
                             *[g[0] for g in gpu], *[g[1] for g in gpu], *codes]); f.flush()
            low_gpu_samples = low_gpu_samples + 1 if any(gpu[int(arms[a])][0] < 512 for a in arms if procs[a].poll() is None) else 0
            if free < 15 or low_gpu_samples >= 3:
                status['state'] = 'stopping_resource_guard'
                status['reason'] = 'disk below 15 GiB' if free < 15 else 'GPU free memory below 512 MiB for three samples'
                save_status()
                for proc in procs.values():
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                break
            time.sleep(30)
    codes = {a: p.wait() for a, p in procs.items()}
    status['exit_codes'] = codes
    if status['state'] != 'stopping_resource_guard':
        status['state'] = 'complete' if all(v == 0 for v in codes.values()) else 'failed'
    save_status()
    sys.exit(0 if status['state'] == 'complete' else 1)
except Exception as error:
    status['state'] = 'monitor_or_launch_failed'
    status['reason'] = repr(error)
    save_status()
    for proc in procs.values():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    for proc in procs.values():
        proc.wait()
    raise
finally:
    for file in files:
        file.close()
