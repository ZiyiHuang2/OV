# RCT V1 closeout: STOP

No credible source-dev gain; stop V1 development and new training. No unseen or independent external generalization claim.

|arm|final mIoU|final mF1|best raw iteration|best mIoU|
|---|---:|---:|---:|---:|
|A|41.5383|54.9175|40000|41.5383|
|C|41.5821|54.9715|40000|41.5821|
|M|41.4744|54.8518|40000|41.4744|
|V|41.4876|54.8670|34999|41.5010|
|R|41.4790|54.8565|40000|41.4790|

V best raw iteration 34999 corresponds to best_checkpoint.json step 35000; final is 40000. Historical Base/C use a different commit; principal resolved protocol, data and initialization matched and a one-image CPU output matched, not a full stochastic replay. Seed 42, batch 4, FP32, AdamW 1e-4, 384 input, 40k cosine; 88754 train and 4700 dev, all 35 classes seen. M/V/R completed with exit code 0.

|comparison|corrected|regressed|net accuracy pp|bootstrap 95% CI pp|positive/negative/tied images|
|---|---:|---:|---:|---|---|
|V-M|1580577|1600055|-0.0073|[-0.2376, +0.2080]|154/141/5|
|R-V|2881518|1657988|+0.4584|[-0.0367, +1.0165]|151/144/5|
|R-A|1717442|1380757|+0.1261|[-0.2643, +0.5367]|154/140/6|

Paired 300-image correctness is not mIoU and the CI does not reflect training randomness. V-M does not establish independent message value and R-V does not establish neighbor-selection benefit. See mechanism_analysis.md for conditional evidence and limitations.

Real CUDA Trainer two-step finite gradients and updates, strict loading, one-image bypass max/mean error zero, checkpoint restore and standalone evaluation were verified; 36 CPU tests passed. Resume RNG/data-order continuity and AMP were not established.

2k Base/C/M/V/R mIoU was 23.8919/23.9392/23.9102/23.8675/23.9029 under its own 2k schedule and is not directly comparable to 40k.

Historical source audit reports no exact train/dev duplicates but 32 near-duplicate candidates remain unchecked, geography is unknown and conflicting-mask image groups remain in training. DLRSD 200 was repeatedly used for development; historical data audit reports split leakage and conflicting masks, so no external evaluation was forced.
