"""Instrument real OVRSTrainer optimizer steps; use only for smoke runs."""
import json
import os
import types
from pathlib import Path
import torch
import train_ovrs

original = train_ovrs.OVRSTrainer.build_optimizer.__func__

def build_optimizer(cls, cfg, model):
    optimizer = original(cls, cfg, model)
    names = {id(p): n for n, p in model.named_parameters() if 'rival_update' in n}
    covered = {id(p) for group in optimizer.param_groups for p in group['params']}
    step = optimizer.step
    records = []
    def audited_step(self, *args, **kwargs):
        params = [(names[id(p)], p) for group in optimizer.param_groups for p in group['params'] if id(p) in names]
        before = {n: p.detach().clone() for n, p in params}
        stats = {n: {'gradient_present': p.grad is not None,
                     'gradient_finite': bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,
                     'gradient_norm': p.grad.float().norm().item() if p.grad is not None else None}
                 for n, p in params}
        for n, p in params:
            if p.grad is not None and n.endswith(('control.2.weight', 'control.2.bias')):
                stats[n]['gate_mix_gradient_norms'] = p.grad.float().reshape(2, -1).norm(dim=1).tolist()
            if p.grad is not None and n.endswith('pair.0.weight'):
                stats[n]['relation_input_gradient_norms'] = p.grad.float().norm(dim=0).tolist()
        result = step(*args, **kwargs)
        for n, p in params:
            stats[n]['update_norm'] = (p.detach() - before[n]).float().norm().item()
        records.append(stats)
        name = 'optimizer_audit_' + os.environ.get('AUDIT_LABEL', 'initial') + '.json'
        Path(cfg.OUTPUT_DIR, name).write_text(json.dumps({'missing_optimizer': [n for pid, n in names.items() if pid not in covered], 'steps': records}, indent=2))
        return result
    optimizer.step = types.MethodType(audited_step, optimizer)
    return optimizer

train_ovrs.OVRSTrainer.build_optimizer = classmethod(build_optimizer)
if os.environ.get('AUDIT_STOP_ITER'):
    from detectron2.engine import HookBase
    class StopAfterCheckpoint(HookBase):
        def after_step(self):
            if self.trainer.iter == int(os.environ['AUDIT_STOP_ITER']):
                raise RuntimeError('AUDIT_STOP_AFTER_CHECKPOINT')
    original_hooks = train_ovrs.OVRSTrainer.build_hooks
    def build_hooks(self):
        return original_hooks(self) + [StopAfterCheckpoint()]
    train_ovrs.OVRSTrainer.build_hooks = build_hooks
args = train_ovrs.make_parser().parse_args()
train_ovrs.launch(train_ovrs.main, args.num_gpus, num_machines=args.num_machines,
                  machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
