# DLRSD semantic-structure pilot, 2026-10-28

Real frozen CLIP ViT-B/16 + RSIB ViT-B/8 extraction and all four engineering checks passed. Fixed seed42, train32/val16, 100 AdamW updates at lr2e-4, batch2, final checkpoint only. This is a cached-view, single-dataset, seen-only engineering pilot, not an HG reproduction or cross-dataset result. The validation subset contains 10 of 14 seen classes; unseen pixels are ignored.

| Arm | Configuration | mIoU % | Effective parameters | Allocated MiB | ms/step |
|---|---|---:|---:|---:|---:|
| B | mean + fusion | 7.365963 | 229696 | 109.32 | 41.58 |
| C | adaptive + fusion | 7.367852 | 229697 | 184.73 | 59.50 |
| D | mean + transport | 7.274434 | 229697 | 297.31 | 48.42 |
| E | adaptive + transport | 7.339052 | 229698 | 296.59 | 70.97 |

C−B=+0.001889pp, D−B=−0.091529pp, E−B=−0.026911pp. No reliable module benefit is established. Predictions are dominated by pavement and trees; the short run is not converged. Do not expand this result into a method claim. Timing is from a shared GPU and memory from a sequential process, not an isolated benchmark. Effective parameters exclude inactive mode scalars and the shared softmax-invariant head bias; registered parameters are 229699 in every arm.

Nine CPU tests, 12 synthetic GPU mode/vocabulary cases, exact real-cache replay, encoder state hashes unchanged, and exact model/optimizer/sampler continuation checks passed. Original nondeterministic runs and failed checks are preserved. The final implementation enables strict deterministic algorithms and flattened-pixel CE; it reran all arms with unchanged samples/settings. Final protocol is in `protocols/dlrsd_pilot_20261028.json`.

Artifacts on myserver: `/home/wangchengjun/huangziyi/semantic_structure_acceptance_20261028`. Full report: `EXECUTION_REPORT.md`; fixed endpoint metrics/checkpoints/logs: `pilot_v2/`; engineering: `engineering_v2/`; cache/provenance: `features_v4/`; executable commands: `RUN_COMMANDS.sh`; source fingerprints: `executed_source_v2/` and `code_identity.json`. No cache, image, mask, encoder weights, checkpoint or predictions are committed.
