# RCT CUDA validation and 2k exploration

Status: IN PROGRESS, 2026-11-15. Five-arm comparison and mechanism analysis are not yet complete.

The upstream experiment source was `bb38fffd79bec940be3fb9d01555f14841060c63`. This work uses the historical OVRSIS95K grouped source train/dev manifest, seed 42, batch 4, FP32, 384 training crop, AdamW, base LR 0.0001 and 2,000-step cosine schedule. All arms select the final checkpoint, evaluate the same 4,700-image source dev set, and use the same full baseline initializer. This is not the historical 40k schedule or an independent external test.

The native A arm completed 2,000 updates and full source-dev evaluation: mIoU 23.8919, macro F1 34.0792. One seed and one arm cannot establish an RCT effect. The full result including per-image RLE and confusion matrix is at `/home/wangchengjun/huangziyi/experiments/rct_2k_bb38fff_20261115/A/`. Its checkpoint SHA-256 is `b93407b7bc9b26b09f845775fe5cc704ae6dbe59f99cf5e6eb563289dd4748c3`.

Full-weight FP32 RCT bypass on one real 35-class image gave max and mean absolute error 0. Real two-update M/V/R smokes had finite gradients, optimizer coverage and actual parameter changes. The individual gate and mix gradient channels, relation input gradients, checkpoint resume with nonzero updates, and repeated eval in one output root were also checked. See `cuda_smoke_report.json` and channel audit JSONs. The small two-image evaluations are engineering checks, not method metrics.

Historical source dev audit reports no exact train/dev image overlap, but geographic independence is unknown and 32 near-duplicate candidate pairs remain unverified. The official val split was excluded due to reported exact train overlap. The 35-class source vocabulary is all seen; unseen mIoU and hIoU are not applicable. DLRSD has repeatedly been used for development and is not a sealed independent test.

Competition C and the M/V/R exploration arms and all paired mechanism analyses remain pending at this checkpoint. Do not make a full-training go/no-go decision from A alone.
