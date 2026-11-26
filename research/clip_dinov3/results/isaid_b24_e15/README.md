# CLIP + DINOv3 C-series on iSAID

All nine training runs and all scheduled evaluations finished on 2026-11-25 at 23:44 China Standard Time. C1 had the highest full-validation mIoU in this run, **81.30%**, versus **78.59%** for the simple CLIP/DINOv3 projection baseline B0.

## Protocol

| Item | Setting |
| --- | --- |
| Training code | `22bec800453a69be9a75515aac970d237ce9b517` |
| Data | OVRSISBench iSAID, 18,076 train and 6,363 validation image/mask pairs with disjoint sample IDs |
| Labels | 15 foreground classes, ignore ID 255 |
| Input and precision | 384×384 encoder input, FP32, TF32 disabled |
| Encoders | Frozen OpenAI CLIP ViT-B/16 vision and text, frozen DINOv3 ViT-B/16 LVD-1689M where applicable |
| Training | Batch 24, 15 epochs, 754 updates per epoch, 11,310 actual updates for each variant, seed 42 |
| Optimizer | AdamW, weight decay 0.01, global gradient clipping 1.0 |
| LR | 300-update linear warmup to 0.001, cosine decay to 0.0001 at update 11,310 |
| C5/C6 | Pseudo supervision disabled. C6 trains with all 15 classes and applies selection at inference |
| Selection and validation | Same seed-42 512-image subset at epochs 5, 10 and 15. Full 6,363-image validation uses the final checkpoint |
| Hardware | Two RTX 4090s running independent single-GPU jobs. Queue A ran C0, D0, C2, C3, C5. Queue B ran B0, C1, C4, C6 |

Dataset audit found 24,439 valid matched RGB / single-channel PNG pairs. Both splits cover all 15 classes at native resolution 256×256. File hashes matched the audited manifest, and the sample ID, scene prefix and image SHA-256 sets were disjoint across train and validation. The dataset audit records the deployed files and the original [published iSAID archive revision](https://huggingface.co/datasets/xiximilu/OVSISBenchDataset/blob/1a29a2e77a76ddb263e1c985452bbcc2bc7f8d91/OVSISBenchDataset/iSAID.zip).

Batch 24 for 15 epochs gives 11,310 optimizer updates versus 11,300 for the planned batch-8, five-epoch run. The 15-epoch runs consume 271,140 training images per variant, counting repeats. All variants keep the final partial batch of four images in each epoch.

The experiment code fixes the dataset gate and report validation, supports the 15-epoch snapshot schedule, and runs independent variants on two GPUs. Rolling recovery checkpoints were saved every 500 updates and at epoch boundaries. Epoch 5/10 evaluation exports and completed recovery states were removed after validation. Final checkpoint hashes are recorded in the evidence bundle.

## Results

All scores are foreground mIoU in percent. The three dev columns are the original report values. The final column contains only results on the full independent validation split.

| Variant | Main change | Dev 5 | Dev 10 | Dev 15 | Full 15 |
| --- | --- | ---: | ---: | ---: | ---: |
| C0 | CLIP projection, bilinear upsampling | 50.88 | 53.38 | 53.80 | 62.67 |
| B0 | CLIP/DINOv3 concatenation and `[I,0]` projection, bilinear upsampling | 72.54 | 73.02 | 73.51 | 78.59 |
| D0 | CLIP semantic seed with dual-guided decoder | 60.04 | 66.90 | 64.75 | 72.95 |
| C1 | B0 seed, multi-layer residual and dual-guided decoder | 73.89 | 76.67 | 76.23 | **81.30** |
| C2 | D0 with DINO/text alignment and contextual score fusion | 72.55 | 67.65 | 74.01 | 80.93 |
| C3 | D0 with relation propagation | 61.71 | 65.87 | 64.33 | 73.90 |
| C4 | C2 with relation propagation | 71.27 | 73.20 | 72.48 | 81.14 |
| C5 | C4 with prototype memory and pixel ranking | 67.30 | 73.62 | 72.53 | 80.23 |
| C6 · default selection | C5 with inference category selection | 72.35 | 75.15 | 73.51 | 80.28 |
| C6 · keep-all | Same C6 final checkpoint with selection bypassed | — | — | — | 80.31 |

The C6 comparison uses the same final checkpoint and the same ordered 6,363 validation images. Default selection retained **10.60 of 15** classes per image on average and reduced mIoU by **0.0303 percentage points**. Mean GPU inference latency was **67.79 ms/image with selection** and **53.35 ms/image with keep-all**, an observed increase of **14.44 ms/image (27.1%)**. These were sequential runs on a shared server. This measurement does not isolate the cost of selection from GPU load.

## Runtime and storage

The two queues took about **7 hours 14 minutes** from formal launch through completion, including evaluation. Training time and throughput below are from each training invocation's wall clock, including data loading and checkpoint writes. The two peak memory columns are PyTorch counters. The last column is the mean CUDA evaluation time per image at batch 1 on the full validation split, with 30 warmup iterations.

| Variant | Training min | Train images/s | Allocated MiB | Reserved MiB | Final checkpoint MiB | Val ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 29.3 | 154.2 | 1,494 | 1,694 | 1.00 | 14.20 |
| B0 | 53.4 | 84.6 | 1,826 | 2,216 | 2.50 | 30.75 |
| D0 | 62.3 | 72.5 | 4,082 | 5,166 | 1.35 | 34.53 |
| C1 | 65.0 | 69.5 | 5,386 | 6,402 | 10.48 | 36.95 |
| C2 | 71.7 | 63.0 | 4,391 | 5,206 | 5.45 | 33.14 |
| C3 | 88.0 | 51.4 | 4,127 | 5,186 | 1.36 | 38.34 |
| C4 | 82.5 | 54.8 | 4,437 | 5,226 | 5.46 | 45.14 |
| C5 | 146.2 | 30.9 | 4,461 | 5,396 | 14.52 | 55.14 |
| C6 · default selection | 133.7 | 33.8 | 4,684 | 5,614 | 14.52 | 67.79 |

The nine final exports total **56.64 MiB**. Frozen encoder weights are shared across runs, with about 662 MiB in the two existing weight files. The preflight server had three 24,564 MiB RTX 4090s, and the formal runs used two. The filesystem had 109–122 GiB available during acceptance sampling.

## Interpretation and decision

- The strongest result is the simple dual-encoder baseline. B0 improves over C0 by **15.92 percentage points**, with positive differences for all 15 classes. C1 adds **2.71 points** over B0 while changing multi-layer fusion and the decoder together. The run supports the combined recipe and leaves the component attribution open.
- D0, C2, C3 and C4 provide a matched factorial comparison. C2 gains **7.98 points** over D0. Relation propagation gains **0.95 points** on D0 and **0.21 points** on C2. The observed interaction is **−0.74 points**. C5 loses **0.91 points** versus C4. These effects come from one seed, and the C5 comparison changes memory and ranking together.
- The fixed 512-image subset has no helicopter ground truth. The evaluator skips classes with empty union, so a helicopter prediction can add a zero-IoU class and change the mean's denominator. On the fixed 14 classes with ground truth, C2's dev curve is **72.55 → 72.48 → 74.01**. The full split covers all 15 classes and remains comparable across runs.
- OVRS ViT-B/16 reports **87.87%** in the separate iSAID validation column of Table II (batch 4, 100,000 iterations) and **93.29%** in the full-benchmark iSAID column. The paper's printed iSAID training count conflicts with the reported total and validation count. The later **94.86%** ViT-L number appears in the RSKT-Seg benchmark table. The public OVRS benchmark script selects `iSAID_all_sem_seg`, whose deployed list includes train and val. The sample list of the exact 94.86% run remains unverified, so the 6,363-image val comparison uses 87.87% as a literature reference. [OVRS paper, Table II](https://arxiv.org/html/2409.07683#S4) · [OVRS evaluation script](https://github.com/caoql98/OVRS/blob/main/eval.sh) · [RSKT-Seg comparison table](https://arxiv.org/html/2509.12040v3)
- B0 is the retained simple baseline and C1 is the best source-validation configuration in this run. Architecture search is paused. The same-vocabulary iSAID experiment does not establish unseen-class or cross-region gains. The archived [vehicle-region omission note](vehicle_omission.md) records two matched cases and the missing strong-baseline pixel predictions. The current decision is to preserve the evidence and defer a new method topic or inference run.

## Preparation and reproducibility

The real-data smoke used 24 train and 16 val images, batch 8 and FP32. All nine variants made 15 updates, interrupted at update 4 and resumed. Target modules received finite gradients and updated, frozen encoders stayed unchanged, and lightweight checkpoints reloaded successfully. C2's text mapping and C3's relation propagation changed logits under diagnostic intervention. C5/C6 updated their prototype memory and reached all 15 classes. C6 used 15 decoder classes in training and 5–9 classes per image in routed smoke evaluation. The smoke showed small CUDA numerical differences between interrupted and uninterrupted C6 runs. On 16 smoke images the routed class sets agreed, and 1 routed and 2 keep-all argmax pixels differed among 1,048,576 pixels per mode. The smoke and formal result metrics use separate manifests.

Batch size was profiled on 192 matched real images for C0, C1 and C6 at batch sizes 8, 16, 24, 32 and 48. The initial shared-GPU probe selected batch 24 as the best combined steady-state time across those three variants. Alternating C1 runs reached 88.5–89.1 images/s at batch 48 and 81.7–84.0 at batch 24. The batch-48 slowdown in the first probe did not reproduce. At batch 8 and batch 24, the five-epoch protocol's full-data preflight observed sampled C1 process peaks of 3,346 and 6,696 MiB, respectively. The 15-epoch protocol passed a separate nine-variant CUDA preflight with three updates and eight evaluation images per variant on the formal manifest. CPU checks included 20 core tests, seven protocol tests and one Bash integration check, plus Python and shell syntax checks.

To reconstruct the full manifest from the extracted release, use `prepare_isaid.py --dataset-root DATA_ROOT --output NEW_MANIFEST_DIR` with `iSAID_split/{train,val}/{images,D2masks}`. The original run used Python 3.10.14, PyTorch 2.5.1+cu121, CUDA 12.1 and the pinned dependencies in the [experiment README](../../README.md). The runner checks the pretrained weight hashes. For a future authorized rerun, from `research/clip_dinov3` set `CSERIES_EPOCHS=15` and optionally `CSERIES_GPU_IDS` to two distinct GPU IDs separated by a comma. Use the same manifest, weights, new output directory and batch in both phases.

```bash
export CSERIES_EPOCHS=15
bash run_isaid_suite.sh preflight "$manifest" "$clip" "$dino" "$output" 24
bash run_isaid_suite.sh main "$manifest" "$clip" "$dino" "$output" 24
```

## Archived evidence

The server run is named `cseries_isaid_b24_e15_20261125`. The archive was checked against its exit marker, resolved schedule and final checkpoint files. Publication checks recomputed 46 evaluation reports from their confusion matrices, verified the 37 formal CSV rows, and matched the full-validation pixel totals against the dataset audit.

- [Formal result table as CSV](evidence/formal_e15_summary.csv)
- [Formal provenance, final checkpoint hashes, 46 main reports and 18 preflight reports](evidence/formal_reports.json), including per-class IoU, confusion matrices, epoch history, route recall and recorded latency
- [Dataset audit](evidence/data_audit.json) with the server-specific path prefix omitted, [batch 8/24 preflight comparison](evidence/formal_preflight_results.json), [batch sweep](evidence/batch_results.json), [C1 alternating retest](evidence/batch_c1_repeat_results.json)
- [Real-data smoke and resume audit](evidence/acceptance.json) and [mechanism intervention checks](evidence/mechanisms.json)
