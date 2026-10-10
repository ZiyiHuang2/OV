"""Preserve initial training records and append one record per invocation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import torch
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger

from .checkpoint import model_config


def repository_state(root):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return {'commit': commit, 'working_tree_dirty': bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'working_tree_dirty': None}


def dataset_summary(manifest_path):
    path = Path(manifest_path).resolve()
    content = path.read_bytes()
    manifest = json.loads(content)
    return {
        'path': str(path), 'manifest_sha256': hashlib.sha256(content).hexdigest(),
        'train_images': len(manifest['train']), 'val_images': len(manifest['val']),
        'seen_class_ids': manifest['seen_class_ids'], 'split_mode': manifest.get('split_mode'),
        'vocabulary_sha256': {
            split: hashlib.sha256((path.parent / manifest[split + '_classes']).read_bytes()).hexdigest()
            for split in ('train', 'test')
        },
    }


def write_run_records(cfg, args, root):
    """Write exclusive per-run files; never replace the initial train snapshot."""
    output = Path(cfg.OUTPUT_DIR)
    mode = 'eval' if args.eval_only else ('resume' if args.resume else 'train')
    outcome = None
    if comm.is_main_process():
        try:
            if mode == 'train' and any((output / name).exists() for name in
                                       ('config.yaml', 'run_provenance.json', 'last_checkpoint', 'metrics.json')):
                raise ValueError('Output already contains training records. Use --resume or a new --output directory.')
            if args.resume and not (output / 'last_checkpoint').is_file():
                raise ValueError('--resume requires an existing checkpoint in the output directory')
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + mode + '_' + uuid.uuid4().hex[:8]
            directory = output / 'runs' / run_id
            directory.mkdir(parents=True, exist_ok=False)
            weights = cfg.MODEL.WEIGHTS
            if args.resume:
                weights = str(output / (output / 'last_checkpoint').read_text().strip())
            provenance = {
                'run_id': run_id, 'mode': mode, 'experiment': args.experiment,
                'weights': weights, 'output_dir': str(output),
                'dataset': dataset_summary(args.dataset_manifest),
                'repository': repository_state(root),
                'source': json.loads((Path(root) / 'docs' / 'UPSTREAM.json').read_text()),
                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'seed': cfg.SEED, 'ovrs': dict(cfg.MODEL.OVRS),
                'model_config': model_config(cfg),
                'allow_unverified_checkpoint': args.allow_unverified_checkpoint,
            }
            with (directory / 'config.yaml').open('x') as file:
                file.write(cfg.dump())
            with (directory / 'run_provenance.json').open('x') as file:
                json.dump(provenance, file, indent=2)
            if mode == 'train':
                # Root snapshots retain their original meaning for existing tools.
                with (output / 'config.yaml').open('x') as file:
                    file.write(cfg.dump())
                with (output / 'run_provenance.json').open('x') as file:
                    json.dump(provenance, file, indent=2)
            outcome = {'run_id': run_id}
        except (OSError, ValueError, KeyError) as error:
            outcome = {'error': str(error)}
    # Broadcast failures too, so other ranks do not wait indefinitely.
    outcome = comm.all_gather(outcome)[0]
    if 'error' in outcome:
        raise ValueError(outcome['error'])
    return output / 'runs' / outcome['run_id']


def setup_run(cfg, args, root):
    directory = write_run_records(cfg, args, root)
    rank = comm.get_rank()
    setup_logger(str(directory), distributed_rank=rank, name='fvcore')
    logger = setup_logger(str(directory), distributed_rank=rank)
    logger.info('Command line arguments: %s', args)
    logger.info('Environment info:\n%s', collect_env_info())
    logger.info('Running with config:\n%s', cfg.dump())
    logger.info('Invocation records: %s', directory)
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)
    if not args.eval_only:
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
    return directory
