#!/usr/bin/env python3
"""CAT-Seg training/evaluation entry point with explicit RS dataset setup."""
import json
import hashlib
import logging
from pathlib import Path
import weakref

import numpy as np
from PIL import Image
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils import comm

from cat_seg import add_cat_seg_config
from cat_seg.data.datasets.register_rs import register_rs_manifest
from cat_seg.engine import Trainer
from cat_seg.engine.checkpoint import EvaluationCheckpointer, OVRSCheckpointer, model_config
from cat_seg.engine.records import setup_run

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('baseline', 'parallel_simple', 'parallel_epl', 'lsk', 'conv',
               'cad', 'epl_lsk', 'epl_cad', 'full', 'full_cad_both',
               'parallel_gated', 'cad_weighted', 'full_selective',
               'full_epl_uniform', 'full_class_cad', 'full_uniform_class_cad',
               'interaction_guidance', 'interaction_fusion', 'interaction_decoder', 'full_interaction')


def harmonic_iou(seen, unseen):
    if not np.isfinite(seen) or not np.isfinite(unseen):
        return float('nan')
    return 2 * seen * unseen / (seen + unseen) if seen + unseen > 0 else 0.0


class SeenUnseenEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, *, collect_samples=False, **kwargs):
        super().__init__(dataset_name, **kwargs)
        metadata = MetadataCatalog.get(dataset_name)
        self.names = metadata.stuff_classes
        self.seen_ids = set(metadata.seen_class_ids)
        self.collect_samples = collect_samples
        self.sample_records = []

    def reset(self):
        super().reset()
        self.sample_records = []

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        if not self.collect_samples:
            return
        k = len(self.names)
        for item, output in zip(inputs, outputs):
            pred = output['sem_seg'].argmax(dim=0).to('cpu').numpy().astype(np.int64)
            with Image.open(self.input_file_to_gt_file[item['file_name']]) as image:
                target = np.asarray(image, dtype=np.int64)
            valid = (target != self._ignore_label)
            hist = np.bincount(k * target[valid] + pred[valid], minlength=k*k).reshape(k, k)
            self.sample_records.append({'file_name': item['file_name'],
                                        'ground_truth_pixels': hist.sum(axis=1).tolist(),
                                        'predicted_pixels': hist.sum(axis=0).tolist(),
                                        'intersection_pixels': np.diag(hist).tolist()})

    def evaluate(self):
        records = comm.all_gather(self.sample_records) if self.collect_samples and self._distributed else [self.sample_records]
        result = super().evaluate()
        if self.collect_samples and result is not None and self._output_dir:
            out = Path(self._output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'per_image.jsonl').open('x') as f:
                for group in records:
                    for row in group:
                        f.write(json.dumps(row) + '\n')
            with (out / 'confusion_matrix.npy').open('xb') as f:
                np.save(f, self._conf_matrix)
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
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        arm = cfg.MODEL.OVRS.COMPETITION_ARM
        if arm:
            from research.ov_competition_v1.install import install
            checkpoint = torch.load(cfg.MODEL.OVRS.COMPETITION_BASE, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg.SEED)
                install(model, arm, initial_scale=cfg.MODEL.OVRS.COMPETITION_SCALE)
            if arm != 'A':
                from research.ov_competition_v1.statistics import attach_statistics
                attach_statistics(model.sem_seg_head.predictor.transformer.competition_update)
        rival = cfg.MODEL.OVRS.RIVAL_ARM
        if rival:
            from research.ov_rival_bridge.install import install as install_rival
            checkpoint = torch.load(cfg.MODEL.OVRS.RIVAL_BASE, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg.SEED)
                install_rival(model, rival, initial_scale=cfg.MODEL.OVRS.RIVAL_SCALE,
                              rank=cfg.MODEL.OVRS.RIVAL_RANK,
                              attn_dim=cfg.MODEL.OVRS.RIVAL_ATTN_DIM)
            from research.ov_rival_bridge.statistics import attach_statistics as attach_rival_statistics
            attach_rival_statistics(model.sem_seg_head.predictor.transformer.rival_update)
        return model

    def __init__(self, cfg, *, run_directory, resume=False, allow_unverified=False):
        self.run_directory = run_directory
        self.resume_requested = resume
        self.allow_unverified = allow_unverified
        super().__init__(cfg)

    def build_hooks(self):
        # Replace before the periodic hook captures its checkpointer reference.
        self.checkpointer = OVRSCheckpointer(
            self.model, self.cfg.OUTPUT_DIR, trainer=weakref.proxy(self),
            expected_config=model_config(self.cfg), strict=self.resume_requested,
            allow_unverified=self.allow_unverified,
            provenance=json.loads((self.run_directory / 'run_provenance.json').read_text()),
        )
        return super().build_hooks()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None and cfg.MODEL.OVRS.COMPETITION_ARM:
            from detectron2.utils.events import get_event_storage
            try:
                event_iter = get_event_storage().iter
                output_folder = str(Path(cfg.OUTPUT_DIR) / 'inference' / f'event_iter_{event_iter:07d}')
            except AssertionError:
                output_folder = str(Path(cfg.OUTPUT_DIR) / 'inference' / 'final')
        return SeenUnseenEvaluator(
            dataset_name, distributed=True,
            collect_samples=bool(cfg.MODEL.OVRS.COMPETITION_ARM),
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
    default_output = 'output/' + args.experiment
    if args.eval_only and not args.resume:
        default_output += '_eval'
    cfg.OUTPUT_DIR = str(Path(args.output or default_output).resolve())
    if args.batch_size is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    if args.max_iter is not None:
        cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.merge_from_list(args.opts)
    if args.competition_arm and args.rival_arm:
        raise ValueError('Competition V1 and rival bridge are mutually exclusive')
    if args.competition_arm:
        cfg.MODEL.OVRS.COMPETITION_ARM = args.competition_arm
        if not args.competition_base_weights:
            raise ValueError('Competition A/B/C requires --competition-base-weights')
        base = Path(args.competition_base_weights).resolve()
        if not base.is_file():
            raise FileNotFoundError(base)
        cfg.MODEL.OVRS.COMPETITION_BASE = str(base)
        with base.open('rb') as f:
            digest = hashlib.sha256()
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
            cfg.MODEL.OVRS.COMPETITION_BASE_SHA256 = digest.hexdigest()
        if not args.eval_only and not args.resume and str(Path(cfg.MODEL.WEIGHTS).resolve()) != str(base):
            raise ValueError('Initial training weights must equal --competition-base-weights')
        if cfg.MODEL.OVRS.AGGREGATION != 'serial' or cfg.MODEL.OVRS.CONTEXT_ADAPTER != 'none' or cfg.MODEL.OVRS.CAD_STAGES or cfg.MODEL.PROMPT_ENSEMBLE_TYPE != 'single':
            raise ValueError('Competition V1 requires native serial aggregation and single prompt')
    elif args.competition_base_weights:
        raise ValueError('--competition-base-weights requires --competition-arm')
    if args.rival_arm:
        cfg.MODEL.OVRS.RIVAL_ARM = args.rival_arm
        if not args.rival_base_weights:
            raise ValueError('Rival bridge requires --rival-base-weights')
        base = Path(args.rival_base_weights).resolve()
        if not base.is_file():
            raise FileNotFoundError(base)
        cfg.MODEL.OVRS.RIVAL_BASE = str(base)
        with base.open('rb') as f:
            digest = hashlib.sha256()
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
            cfg.MODEL.OVRS.RIVAL_BASE_SHA256 = digest.hexdigest()
        if not args.eval_only and not args.resume and str(Path(cfg.MODEL.WEIGHTS).resolve()) != str(base):
            raise ValueError('Rival training weights must equal --rival-base-weights')
        if (cfg.MODEL.OVRS.AGGREGATION != 'serial' or
                cfg.MODEL.OVRS.CONTEXT_ADAPTER != 'none' or
                cfg.MODEL.OVRS.CAD_STAGES or
                cfg.MODEL.PROMPT_ENSEMBLE_TYPE != 'single'):
            raise ValueError('Rival bridge requires native serial aggregation and single prompt')
        if cfg.MODEL.OVRS.RIVAL_RANK < 1 or cfg.MODEL.OVRS.RIVAL_ATTN_DIM < 1:
            raise ValueError('Rival rank and attention dimensions must be positive')
    elif args.rival_base_weights:
        raise ValueError('--rival-base-weights requires --rival-arm')
    cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR).resolve())
    if cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED != 'ViT-B/16':
        raise ValueError('This RS runner supports the tested OpenAI CLIP ViT-B/16 path')
    if cfg.SOLVER.IMS_PER_BATCH < max(args.num_gpus, 1) or cfg.SOLVER.IMS_PER_BATCH % max(args.num_gpus, 1):
        raise ValueError('Global batch size must be divisible by number of GPUs')
    cfg.freeze()
    return cfg, setup_run(cfg, args, ROOT)


def main(args):
    cfg, run_directory = setup(args)
    torch.set_float32_matmul_precision('high')
    if args.eval_only:
        if not cfg.MODEL.WEIGHTS and not args.resume:
            raise ValueError('Evaluation requires --weights or --resume')
        model = OVRSTrainer.build_model(cfg)
        checkpointer = EvaluationCheckpointer(
            model, save_dir=cfg.OUTPUT_DIR, expected_config=model_config(cfg),
            allow_unverified=args.allow_unverified_checkpoint,
        )
        if not cfg.MODEL.WEIGHTS and not (args.resume and checkpointer.has_checkpoint()):
            raise ValueError('No evaluation checkpoint: supply --weights or an existing --resume output')
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume,
        )
        evaluators = [OVRSTrainer.build_evaluator(
            cfg, name, output_folder=str(run_directory / 'inference' / name),
        ) for name in cfg.DATASETS.TEST]
        result = OVRSTrainer.test(cfg, model, evaluators)
        if comm.is_main_process():
            with (run_directory / 'evaluation.json').open('x') as file:
                json.dump(result, file, indent=2)
        return result
    trainer = OVRSTrainer(
        cfg, run_directory=run_directory, resume=args.resume,
        allow_unverified=args.allow_unverified_checkpoint,
    )
    if args.resume and not trainer.checkpointer.has_checkpoint():
        raise ValueError('--resume requires an existing checkpoint in the output directory')
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def make_parser():
    parser = default_argument_parser()
    parser.add_argument('--dataset-manifest', required=True)
    parser.add_argument('--experiment', choices=EXPERIMENTS, default='baseline')
    parser.add_argument('--weights')
    parser.add_argument('--competition-arm', choices=('A', 'B', 'C'))
    parser.add_argument('--competition-base-weights')
    parser.add_argument('--rival-arm', choices=('S', 'M', 'R', 'P'))
    parser.add_argument('--rival-base-weights')
    parser.add_argument('--output')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-iter', type=int)
    parser.add_argument('--allow-unverified-checkpoint', action='store_true',
                        help='Explicitly permit metadata-free legacy weights for evaluation/resume')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
