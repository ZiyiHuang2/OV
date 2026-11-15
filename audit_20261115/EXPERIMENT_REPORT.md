# RCT CUDA validation and 2k exploration

Status: IN PROGRESS, 2026-11-15. Five-arm comparison and mechanism analysis are not yet complete.

The upstream experiment source was `bb38fffd79bec940be3fb9d01555f14841060c63`. This work uses the historical OVRSIS95K grouped source train/dev manifest, seed 42, batch 4, FP32, 384 training crop, AdamW, base LR 0.0001 and 2,000-step cosine schedule. All arms select the final checkpoint, evaluate the same 4,700-image source dev set, and use the same full baseline initializer. This is not the historical 40k schedule or an independent external test.

Native A and Competition C completed 2,000 updates and full source-dev evaluation: A mIoU 23.8919, macro F1 34.0792; C mIoU 23.9392, macro F1 34.1250. The C minus A mIoU difference is +0.0473 percentage points. These are single-seed 2k results, not 40k training or evidence that RCT is effective. The full results including per-image RLE and confusion matrices are under `/home/wangchengjun/huangziyi/experiments/rct_2k_bb38fff_20261115/`. Checkpoint SHA-256 values: A `b93407b7bc9b26b09f845775fe5cc704ae6dbe59f99cf5e6eb563289dd4748c3`, C `7bd942f14d0770d31d2549e1fe994f1a7648c270177a163a806e065e1a30367b`.

Full-weight FP32 RCT bypass on one real 35-class image gave max and mean absolute error 0. Real two-update M/V/R smokes had finite gradients, optimizer coverage and actual parameter changes. The individual gate and mix gradient channels, relation input gradients, checkpoint resume with nonzero updates, and repeated eval in one output root were also checked. See `cuda_smoke_report.json` and channel audit JSONs. The small two-image evaluations are engineering checks, not method metrics.

Historical source dev audit reports no exact train/dev image overlap, but geographic independence is unknown and 32 near-duplicate candidate pairs remain unverified. The official val split was excluded due to reported exact train overlap. The 35-class source vocabulary is all seen; unseen mIoU and hIoU are not applicable. DLRSD has repeatedly been used for development and is not a sealed independent test.

M and V are training concurrently on GPU 0 and GPU 1; R will follow V. GPU 1 is shared, so throughput is not an isolated benchmark. RCT exploration results and paired mechanism analyses remain pending. Do not make a full-training go/no-go decision from A/C alone.
