# iSAID vehicle-region omission

Status on 2026-11-26 is **archived lead, method topic deferred**. This is a read-only comparison of existing predictions.

The old SegEarth-OV and Pi Full diagnostic archives contain two predictions aligned to the same iSAID validation masks. Each row below is an 8-connected large-vehicle ground-truth region. Coverage is the share of region pixels predicted as either small or large vehicle.

| Full crop ID | Region ID | GT pixels | SegEarth-OV | Old Pi Full |
| --- | ---: | ---: | ---: | ---: |
| `P1982_206_462_1236_1492` | 3 | 2,403 | 0.00% | 0.00% |
| `P1982_206_462_1236_1492` | 6 | 2,304 | 0.00% | 0.00% |
| `P1476_618_874_1236_1492` | 4 | 4,333 | 4.11% | 9.42% |
| `P1476_618_874_1236_1492` | 1 | 3,629 | 0.00% | 9.26% |

In the 38 diagnostic crops matched to the independent iSAID validation split, both predictors assigned a vehicle class to less than 10% of the same region for 20 of 51 large-vehicle regions across five source scenes, and 16 of 50 small-vehicle regions across nine source scenes. These are selected diagnostic crops and connected regions. They provide neither a population miss rate nor a vehicle instance count.

The archive is `SEGEARTH_OV_427_20260927T012700Z`. SegEarth-OV uses `05_run427/exports/iSAID/tensors/<crop_id>.pt`. Old Pi Full uses `08_pi_full_comparator/run427/iSAID/<crop_id>.pt`, with PiNI enabled, pool `[2,2]` and checkpoint SHA-256 `de4acf761046b69499a3a484cf4ed512e01e349f165ffc18cd7866ae10b3e763`. Both crops are in the independent validation split. Both comparisons read the saved `final_mask` and verify `gt_eval_mask` against the independent validation mask. The old diagnostic runs and C-series use different training and evaluation protocols.

**Missing evidence.** No saved pixel-level predictions for the two crops were located for B0, C1 or another strong baseline with the same training and evaluation protocol. C-series retained checkpoint and aggregate evaluation reports. Those artifacts do not resolve whether the regions remain omitted. Evidence of the recurrence and downstream cost in a real mapping workflow, or a link to genuinely unseen vocabulary, is also missing. The current decision is to preserve the observation and defer inference and topic selection.
