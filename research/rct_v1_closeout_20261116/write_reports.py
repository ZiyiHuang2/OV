import csv,json,hashlib
from pathlib import Path
p=Path('/home/wangchengjun/huangziyi/experiments/rct_v1_closeout_20261116')
R=list(csv.DictReader((p/'results_40k.csv').open())); S=list(csv.DictReader((p/'paired_full_pixel_summary.csv').open())); G=list(csv.DictReader((p/'paired_analysis.csv').open()))
table='|arm|final mIoU|final mF1|best raw iteration|best mIoU|\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"|{x['arm']}|{float(x['final_miou']):.4f}|{float(x['final_macro_f1']):.4f}|{x['best_iter']}|{float(x['best_miou']):.4f}|" for x in R)
paired='|comparison|corrected|regressed|net accuracy pp|bootstrap 95% CI pp|positive/negative/tied images|\n|---|---:|---:|---:|---|---|\n'+'\n'.join(f"|{x['comparison']}|{x['corrected']}|{x['regressed']}|{float(x['net_pp']):+.4f}|[{float(x['ci_low_pp']):+.4f}, {float(x['ci_high_pp']):+.4f}]|{x['positive_images']}/{x['negative_images']}/{x['tied_images']}|" for x in S)
st='|comparison|Base stratum|corrected|regressed|net pp|95% CI pp|\n|---|---|---:|---:|---:|---|\n'+'\n'.join(f"|{x['comparison']}|{x['stratum']}|{x['corrected']}|{x['regressed']}|{float(x['net_pp']):+.4f}|[{float(x['bootstrap_low_pp']):+.4f},{float(x['bootstrap_high_pp']):+.4f}]|" for x in G if x['stratum'] in ['low_margin','high_margin','switch','stable','gt_top3','gt_not_top3'])
(p/'mechanism_analysis.md').write_text('# RCT V1 paired mechanism analysis\n\n'+paired+'\n\nCommon Base coarse-grid regions:\n\n'+st+"""
Frozen first 300 manifest entries, prior to outcome inspection. Margin threshold 0.048815809190273285 is the pooled valid Base top1 minus top2 median. Switch is a change in Base runner-up identity at any valid 8-neighbor, not model-specific. GT rank is from Base raw scores. The 24x24 labels and predictions use nearest-neighbor mapping after mapper resize and padding; these are coarse cells, not all image pixels. Switch covers 156081/172800 cells (90.3%), so it is not a selective boundary-only set.

Full-pixel CIs use 2000 image bootstrap replicates seed 42 on pooled net correctness divided by valid pixels, not mIoU and not seed uncertainty. No principal V-M or R-V regional CI establishes reliable benefit. R-V full-pixel estimate is positive but CI spans zero. Positive and negative image counts are similar. Trained checkpoints also have different optimization trajectories; these are not isolated runtime ablations.

R-V corrects GT grass tree->grass 987996 and GT tree grass->tree 538914, but changes correct GT tree tree->grass 417387 and correct GT road road->grass 198585. V-M changes correct grass grass->tree 390369 and corrects tree grass->tree 373830. error_transitions.csv includes wrong-to-wrong transitions; paired_class_summary.csv and paired_class_per_image.csv preserve class effects.

Coverage: 34/35 GT classes present; windmill absent, train station appears in one image and dam in six. On the coarse grid they appear in one and four images. Rare-class bootstrap intervals are not reliable. No verified scene labels are available; scene representativeness is unknown. Manifest-order bias is possible; selection was not changed.

Prior 100-image bypass measures dependence of a co-trained system, not superiority to independent Base. Alternate rival holds the old margin fixed, creating inconsistent conditions; R also changes same_rival. Subset mIoU is not full 4700-image mIoU. Incorrect rival selection, spatial switching, host redundancy and coadaptation/optimization remain unproven explanations.
""",encoding='utf8')
(p/'EXPERIMENT_CLOSEOUT.md').write_text('# RCT V1 closeout: STOP\n\nNo credible source-dev gain; stop V1 development and new training. No unseen or independent external generalization claim.\n\n'+table+'\n\nV best raw iteration 34999 corresponds to best_checkpoint.json step 35000; final is 40000. Historical Base/C use a different commit; principal resolved protocol, data and initialization matched and a one-image CPU output matched, not a full stochastic replay. Seed 42, batch 4, FP32, AdamW 1e-4, 384 input, 40k cosine; 88754 train and 4700 dev, all 35 classes seen. M/V/R completed with exit code 0.\n\n'+paired+"""

Paired 300-image correctness is not mIoU and the CI does not reflect training randomness. V-M does not establish independent message value and R-V does not establish neighbor-selection benefit. See mechanism_analysis.md for conditional evidence and limitations.

Real CUDA Trainer two-step finite gradients and updates, strict loading, one-image bypass max/mean error zero, checkpoint restore and standalone evaluation were verified; 36 CPU tests passed. Resume RNG/data-order continuity and AMP were not established.

2k Base/C/M/V/R mIoU was 23.8919/23.9392/23.9102/23.8675/23.9029 under its own 2k schedule and is not directly comparable to 40k.

Historical source audit reports no exact train/dev duplicates but 32 near-duplicate candidates remain unchecked, geography is unknown and conflicting-mask image groups remain in training. DLRSD 200 was repeatedly used for development; historical data audit reports split leakage and conflicting masks, so no external evaluation was forced.
""",encoding='utf8')
(p/'FAILURE_REVIEW.md').write_text("""# Failure review

RCT V1 did not establish source-dev improvement. Value or rival-conditioned selection did not show reliable shared-stratum gains. V/R change predictions and co-trained systems depend on their modules, but this does not establish benefit over Base. Class and image gains and losses coexist.

Potential causes, not proven: inaccurate rival choice, switching, redundant host aggregation, optimization or coadaptation. One seed, all-seen source dev, class imbalance, ordering and missing geographic independence restrict inference. A new V1 training budget is not justified. Future unrelated OV work should specify falsifiable hypotheses and matched controls, but this closeout does not authorize or initiate V2.
""",encoding='utf8')
(p/'REPRODUCIBILITY.md').write_text("""# Reproducibility

Server root: /home/wangchengjun/huangziyi/experiments/rct_v1_closeout_20261116. Repo: /home/wangchengjun/huangziyi/experiments/rct_40k_20261115/repo, commit 16ab4ddd410d46ef31349bca717335bbb5638605 (same tree as PR 5 head fe6295e8d1d951660aac31bce22684493378cf62). Historical Base/C commit 4308cc137bbf8220a3fb0e8c9ce2e44035c06a6f. Python /home/wangchengjun/miniconda3/envs/ovrs/bin/python, PyTorch 2.5.1+cu121; Detectron2 1e3e13bbf607b54f62205c4c33922521822fb298. Inference FP32 and torch float32 matmul precision high.

Paths and checkpoint hashes: checkpoint_manifest.json. Frozen IDs and rules: diagnostic_plan.json; threshold: frozen_threshold.json. From the repo set CUDA_VISIBLE_DEVICES=1, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 and PYTHONPATH=., then invoke the Python above on closeout/infer.py with A, M, V and R, followed by analyze.py and class_analysis.py. Inference refuses to overwrite predictions. To repeat safely, copy scripts, plan and checkpoint manifest to a fresh directory and change the hard-coded closeout root in each script. No large weights or predictions are committed.
""",encoding='utf8')
(p/'EVIDENCE_INDEX.md').write_text("""# Evidence index

- Results: results_40k.csv; originals <arm>/metrics.json and inference/event_iter_0040000/confusion_matrix.npy under /home/wangchengjun/huangziyi/experiments/rct_40k_20261115 for M/V/R and /home/wangchengjun/huangziyi/ov_competition_v1_20261030_audit/formal_20261101_seed42 for A/C.
- Weights, SHA256, config and repo commit: checkpoint_manifest.json; initializer SHA256 a201a50f94f4c04196e3083773832b736c2b8b81d27ea2c59cc13af3aff05760; manifest ccbd48a21ebd97e713ee6ebb001aa69ff50971f51a76d1320f6da0577cc40e3d; class vocabulary 0073371b8450608a1702ecab22939fc60ba24222d2dea1e434047a100e463f41.
- Frozen protocol diagnostic_plan.json and frozen_threshold.json; inference predictions/{A,M,V,R}, infer_*.log and infer.py; full-pixel paired_per_image.csv, paired_full_pixel_summary.csv, paired_class_per_image.csv, paired_class_summary.csv and error_transitions.csv; grid strata strata_per_image.csv and paired_analysis.csv; scripts analyze.py and class_analysis.py.
- Prior 40k evidence /home/wangchengjun/huangziyi/experiments/rct_40k_20261115/{acceptance_report.json,formal_status.json,historical_reference_audit.json}; 40k prior bypass analysis_40k/{M,V,R}.json. 2k evidence /home/wangchengjun/huangziyi/experiments/rct_2k_bb38fff_20261115/{EXPERIMENT_REPORT.md,cuda_smoke_report.json,dataset_audit.json}.
- DLRSD resource audit /home/wangchengjun/huangziyi/ovrs_formal_20261011_dlrsd_v1/data_audit.json. Small-artifact SHA256 in artifact_sha256.json.
""",encoding='utf8')
files=[x for x in p.iterdir() if x.is_file() and x.name!='artifact_sha256.json']
(p/'artifact_sha256.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(files)},indent=2))
print('reports complete')
