#!/usr/bin/env python3
"""CAT-Seg training/evaluation entry point with explicit RS dataset setup."""
import json
import logging
from pathlib import Path

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils import comm

from cat_seg import add_cat_seg_config
from cat_seg.data.datasets.register_rs import register_rs_manifest
from cat_seg.engine import Trainer

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('baseline', 'parallel_simple', 'parallel_epl', 'lsk', 'conv',
               'cad', 'epl_lsk', 'epl_cad', 'full', 'full_cad_both',
               'parallel_gated', 'cad_weighted', 'full_selective')


def harmonic_iou(seen, unseen):
    if not np.isfinite(seen) or not np.isfinite(unseen):
        return float('nan')
    return 2 * seen * unseen / (seen + unseen) if seen + unseen > 0 else 0.0


class EvaluationCheckpointer(DetectionCheckpointer):
    """Training can initialize new modules; evaluation must load every module."""

    def _load_model(self, checkpoint):
        incompatible = super()._load_model(checkpoint)
        if incompatible.missing_keys or incompatible.unexpected_keys or incompatible.incorrect_shapes:
            raise ValueError(
                'Checkpoint does not match the evaluation architecture. Use the same '
                'experiment and module settings as training. '
                f'missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}, '
                f'incorrect_shapes={incompatible.incorrect_shapes}'
            )
        return incompatible


class SeenUnseenEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name, **kwargs)
        metadata = MetadataCatalog.get(dataset_name)
        self.names = metadata.stuff_classes
        self.seen_ids = set(metadata.seen_class_ids)

    def evaluate(self):
        result = super().evaluate()
        if result is None:
            return None
        metrics = result['sem_seg']
        def mean(ids):
            values = [metrics.get('IoU-' + self.names[i], float('nan')) for i in ids]
            values = [value for value in values if np.isfinite(value)]
            return float(np.mean(values)) if values else float('nan')
        seen = mean(sorted(self.seen_ids))
        unseen = mean([i for i in range(len(self.names)) if i not in self.seen_ids])
        metrics['seen_mIoU'] = seen
        metrics['unseen_mIoU'] = unseen
        metrics['hIoU'] = harmonic_iou(seen, unseen)
        logging.getLogger(__name__).info('Seen/unseen metrics: %s', metrics)
        return result


class OVRSTrainer(Trainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return SeenUnseenEvaluator(
            dataset_name, distributed=True,
            output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / 'inference'),
        )


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    config = args.config_file or str(ROOT / 'configs' / 'ovrs' / (args.experiment + '.yaml'))
    cfg.merge_from_file(config)
    registered = register_rs_manifest(args.dataset_manifest)
    cfg.DATASETS.TRAIN = (registered['train_name'],)
    cfg.DATASETS.TEST = (registered['test_name'],)
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = registered['train_classes']
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = registered['test_classes']
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = registered['num_train_classes']
    if args.weights is not None:
        cfg.MODEL.WEIGHTS = str(Path(args.weights).resolve())
    cfg.OUTPUT_DIR = str(Path(args.output or ('output/' + args.experiment)).resolve())
    if args.batch_size is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    if args.max_iter is not None:
        cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED != 'ViT-B/16':
        raise ValueError('This RS runner supports the tested OpenAI CLIP ViT-B/16 path')
    if cfg.SOLVER.IMS_PER_BATCH < max(args.num_gpus, 1) or cfg.SOLVER.IMS_PER_BATCH % max(args.num_gpus, 1):
        raise ValueError('Global batch size must be divisible by number of GPUs')
    cfg.freeze()
    default_setup(cfg, args)
    if comm.is_main_process():
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        provenance = {
            'dataset_manifest': str(Path(args.dataset_manifest).resolve()),
            'experiment': args.experiment, 'weights': cfg.MODEL.WEIGHTS,
            'source': json.loads((ROOT / 'docs' / 'UPSTREAM.json').read_text()),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'seed': cfg.SEED, 'ovrs': dict(cfg.MODEL.OVRS),
        }
        (Path(cfg.OUTPUT_DIR) / 'run_provenance.json').write_text(json.dumps(provenance, indent=2))
    return cfg


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision('high')
    if args.eval_only:
        if not cfg.MODEL.WEIGHTS and not args.resume:
            raise ValueError('Evaluation requires --weights or --resume')
        model = OVRSTrainer.build_model(cfg)
        checkpointer = EvaluationCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        if not cfg.MODEL.WEIGHTS and not (args.resume and checkpointer.has_checkpoint()):
            raise ValueError('No evaluation checkpoint: supply --weights or an existing --resume output')
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume,
        )
        result = OVRSTrainer.test(cfg, model)
        if comm.is_main_process():
            (Path(cfg.OUTPUT_DIR) / 'evaluation.json').write_text(json.dumps(result, indent=2))
        return result
    trainer = OVRSTrainer(cfg)
    if args.resume and not trainer.checkpointer.has_checkpoint():
        raise ValueError('--resume requires an existing checkpoint in the output directory')
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def make_parser():
    parser = default_argument_parser()
    parser.add_argument('--dataset-manifest', required=True)
    parser.add_argument('--experiment', choices=EXPERIMENTS, default='baseline')
    parser.add_argument('--weights')
    parser.add_argument('--output')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-iter', type=int)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
