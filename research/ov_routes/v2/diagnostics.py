"""Post-hoc selector diagnostics; strictly distinct from GT-free inference.

All gains in this module come from SOURCE TRAIN masks and are used only on
source-internal holdout samples, never as features or model-selection labels for
external DLRSD validation.  Per-image GT-present mIoU is a *diagnostic proxy*;
it is not equivalent to an aggregate confusion-matrix mIoU.
"""
from __future__ import annotations

import math
from collections import Counter

import torch


def gt_present_miou(pred: torch.Tensor, labels: torch.Tensor, classes: int) -> torch.Tensor:
    """IoU averaged over GT-present classes (same class subset for all actions)."""
    valid = labels != 255
    if not bool(valid.any()):
        raise ValueError('No supervised pixels for per-image IoU')
    gt, chosen = labels[valid].long(), pred[valid].long()
    if ((gt < 0) | (gt >= classes)).any() or ((chosen < 0) | (chosen >= classes)).any():
        raise ValueError('Invalid class label in IoU diagnostic')
    cm = torch.bincount(gt * classes + chosen, minlength=classes * classes).reshape(classes, classes)
    tp = cm.diag()
    gt_count, pred_count = cm.sum(1), cm.sum(0)
    union = gt_count + pred_count - tp
    keep = gt_count > 0
    return (tp[keep].double() / union[keep].clamp_min(1).double()).mean()


def _correlation(x: torch.Tensor, y: torch.Tensor):
    """Pearson r; return None for degenerate/constant targets."""
    x, y = x.double().flatten(), y.double().flatten()
    if len(x) < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return None
    a, b = x - x.mean(), y - y.mean()
    denom = a.norm() * b.norm()
    return None if float(denom) < 1e-12 else float((a * b).sum() / denom)


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """Average ranks with exact ties; no scipy dependency."""
    values, inverse, counts = torch.unique(x.double().flatten(), sorted=True,
                                            return_inverse=True, return_counts=True)
    del values
    start = torch.cumsum(counts, 0) - counts
    ranks = start.double() + (counts.double() + 1) / 2
    return ranks[inverse]


def _spearman(x: torch.Tensor, y: torch.Tensor):
    return _correlation(_ranks(x), _ranks(y))


def ranking_pair_counts(gains: torch.Tensor, gain_scale: float = 100.0,
                        min_gap: float = .02):
    """Pair support for the existing rank loss; uses the same scaled units."""
    if gains.ndim != 2:
        raise ValueError('Expected [B,A] gains')
    target = gains * gain_scale
    diffs = (target.unsqueeze(-1) - target.unsqueeze(-2)).abs()
    upper = torch.triu(torch.ones_like(diffs, dtype=torch.bool), diagonal=1)
    return {'eligible': int((upper & (diffs > min_gap)).sum()),
            'possible': int(upper.sum())}


def summarize_predictions(rows: list[dict], *, scale: float = 100.0):
    """Summarize GT-based evidence *after* scoring, not during selection.

    Each row needs `predicted` [A] scaled BCE benefit, `gains` [A] raw BCE
    benefit, and optional `iou_gains` [A] GT-present per-image mIoU change.
    A zero-gain no-op can be included in metrics but is not treated as a crop.
    """
    if not rows:
        raise ValueError('No diagnostic records')
    all_pred, all_bce, all_iou = [], [], []
    chosen_bce, chosen_iou = [], []
    oracle_bce, oracle_iou = [], []
    bce_best_harms_iou = 0
    bce_iou_comparable = 0
    hit, pos, can_pos, missed_pos, total = 0, 0, 0, 0, 0
    wrong_sign = 0
    window_counts: Counter = Counter()
    per_image_rank = []
    best_action_iou_gain = []
    for row in rows:
        p = torch.as_tensor(row['predicted'], dtype=torch.float64).flatten()
        g = torch.as_tensor(row['gains'], dtype=torch.float64).flatten()
        if p.numel() != g.numel() or not p.numel() or not torch.isfinite(p).all() or not torch.isfinite(g).all():
            raise ValueError('Nonfinite/mismatched gains')
        selected = int(p.argmax())
        best = float(g.max())
        total += 1
        hit += int(float(g[selected]) >= best-1e-10)
        chosen_bce.append(float(g[selected]))
        oracle_bce.append(best)
        pos += int(float(g[selected]) > 0)
        can_pos += int(best > 0)
        missed_pos += int(best > 0 and float(g[selected]) <= 0)
        window_counts[selected] += 1
        wrong_sign += int((p[selected] > 0) != (g[selected] > 0))
        all_pred.extend(p.tolist())
        all_bce.extend((g*scale).tolist())
        per_image_rank.append(_spearman(p,g))
        if 'iou_gains' in row:
            ig = torch.as_tensor(row['iou_gains'], dtype=torch.float64).flatten()
            if ig.shape != g.shape or not torch.isfinite(ig).all():
                raise ValueError('Invalid IoU gains')
            all_iou.extend(ig.tolist())
            chosen_iou.append(float(ig[selected]))
            idx_bce = int(g.argmax())
            idx_iou = int(ig.argmax())
            oracle_iou.append(float(ig[idx_iou]))
            best_action_iou_gain.append(float(ig[idx_bce]))
            bce_iou_comparable += 1
            bce_best_harms_iou += int(float(ig[idx_bce]) < -1e-12)
    pred, truth = torch.tensor(all_pred), torch.tensor(all_bce)
    valid_r = [x for x in per_image_rank if x is not None]
    summary = {
        'n_images':total, 'n_actions':len(rows[0]['gains']),
        'global_bce_pearson':_correlation(pred,truth),
        'global_bce_spearman':_spearman(pred,truth),
        'mean_imagewise_bce_spearman':sum(valid_r)/len(valid_r) if valid_r else None,
        'top1_bce_oracle_hit_rate':hit/total,
        'chosen_positive_bce_rate':pos/total,
        'images_with_positive_bce_action':can_pos,
        'missed_positive_bce_action':missed_pos,
        'selected_nonpositive_bce_count':total-pos,
        'selected_sign_mismatch_rate':wrong_sign/total,
        'mean_chosen_bce_gain':sum(chosen_bce)/total,
        'mean_best_bce_gain':sum(oracle_bce)/total,
        'mean_bce_regret':sum(a-b for a,b in zip(oracle_bce,chosen_bce))/total,
        'window_counts':{str(i):window_counts[i] for i in range(len(rows[0]['gains']))},
        'ranking_pairs':ranking_pair_counts(torch.tensor([r['gains'] for r in rows],dtype=torch.float64)),
        'metric_scope':'SOURCE internal holdout; per-image actions; not aggregate DLRSD mIoU',
    }
    if bce_iou_comparable == total:
        summary['gt_present_iou'] = {
            'mean_chosen_gain':sum(chosen_iou)/total,
            'mean_oracle_gain':sum(oracle_iou)/total,
            'mean_iou_regret':sum(a-b for a,b in zip(oracle_iou,chosen_iou))/total,
            'bce_best_hurts_iou_rate':bce_best_harms_iou/total,
            'bce_best_hurts_iou_count':bce_best_harms_iou,
            'mean_iou_gain_of_bce_best':sum(best_action_iou_gain)/total,
            'global_bce_vs_iou_spearman':_spearman(torch.tensor(all_bce),torch.tensor(all_iou)),
            'note':'GT-present per-image mIoU proxy, not global dataset mIoU or original 200-image IoU Oracle.'
        }
    return summary


@torch.no_grad()
def evaluate_cached_selector(model, samples: list[dict], device: str):
    model.eval()
    rows=[]
    for sample in samples:
        evidence={k:v.unsqueeze(0).to(device) for k,v in sample['evidence'].items()}
        p=model(evidence)[0].cpu()
        row={'predicted':p.tolist(),'gains':sample['gains'].cpu().tolist(),
             'train_index':sample['meta'].get('train_index'),
             'augmentation':sample['meta'].get('augmentation')}
        if 'iou_gains' in sample:
            row['iou_gains']=sample['iou_gains'].cpu().tolist()
        rows.append(row)
    return summarize_predictions(rows)


def summarize_target_alignment(samples: list[dict]):
    """GT-only *source-cache* diagnostic BEFORE training any selector.

    Computes whether minimizing BCE is a useful proxy for selecting an action
    that improves GT-present per-image IoU.  Does not train a model, touch
    external val or calculate global aggregated DLRSD mIoU.
    """
    if not samples:
        raise ValueError('No cached samples for supervision diagnostic')
    if any('iou_gains' not in s for s in samples):
        raise ValueError('Recompute cache with diagnostic iou_gains included')
    bce = torch.stack([s['gains'].float() for s in samples])
    iou = torch.stack([s['iou_gains'].float() for s in samples])
    if bce.shape != iou.shape or not torch.isfinite(bce).all() or not torch.isfinite(iou).all():
        raise ValueError('Invalid cached action gains')
    n,a = bce.shape
    bce_idx=bce.argmax(1)
    iou_idx=iou.argmax(1)
    row=torch.arange(n)
    bce_chosen_iou=iou[row,bce_idx]
    iou_best=iou[row,iou_idx]
    valid = [v for x,y in zip(bce,iou) if (v:=_spearman(x,y)) is not None]
    cost = ranking_pair_counts(bce)
    result = {
        'n_images_augmented':n, 'n_actions':a,
        'global_bce_iou_pearson':_correlation(bce,iou),
        'global_bce_iou_spearman':_spearman(bce,iou),
        'mean_imagewise_bce_iou_spearman':sum(valid)/len(valid) if valid else None,
        'top1_bce_iou_same_window_rate':float((bce_idx==iou_idx).float().mean()),
        'bce_oracle_selected_iou_harm_count':int((bce_chosen_iou < -1e-12).sum()),
        'bce_oracle_selected_iou_harm_rate':float((bce_chosen_iou < -1e-12).float().mean()),
        'mean_iou_gain_of_bce_best':float(bce_chosen_iou.mean()),
        'mean_iou_gain_of_iou_best':float(iou_best.mean()),
        'mean_iou_regret_of_bce_best':float((iou_best-bce_chosen_iou).mean()),
        'has_any_positive_bce_action_rate':float((bce.max(1).values>0).float().mean()),
        'has_any_positive_iou_action_rate':float((iou.max(1).values>0).float().mean()),
        'bce_positive_but_iou_negative_action_rate':float(((bce>0)&(iou<0)).float().mean()),
        'rank_pairs':cost,
        'bce_best_counts':{str(i):int((bce_idx==i).sum()) for i in range(a)},
        'iou_best_counts':{str(i):int((iou_idx==i).sum()) for i in range(a)},
        'scope':'SOURCE TRAIN cache only; IoU means GT-present per-image class IoU; not global mIoU',
        'no_gt_in_inference':True,
    }
    return result
