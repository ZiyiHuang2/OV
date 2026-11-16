# RCT V1 paired mechanism analysis

|comparison|corrected|regressed|net accuracy pp|bootstrap 95% CI pp|positive/negative/tied images|
|---|---:|---:|---:|---|---|
|V-M|1580577|1600055|-0.0073|[-0.2376, +0.2080]|154/141/5|
|R-V|2881518|1657988|+0.4584|[-0.0367, +1.0165]|151/144/5|
|R-A|1717442|1380757|+0.1261|[-0.2643, +0.5367]|154/140/6|

Common Base coarse-grid regions:

|comparison|Base stratum|corrected|regressed|net pp|95% CI pp|
|---|---|---:|---:|---:|---|
|V-M|low_margin|680|604|+0.0880|[-0.2050,+0.3983]|
|V-M|high_margin|397|401|-0.0046|[-0.1799,+0.1364]|
|V-M|switch|1032|963|+0.0442|[-0.1806,+0.2640]|
|V-M|stable|45|42|+0.0179|[-0.1761,+0.1719]|
|V-M|gt_top3|856|839|+0.0116|[-0.2077,+0.2204]|
|V-M|gt_not_top3|221|166|+0.2118|[-0.0341,+0.4691]|
|R-V|low_margin|732|593|+0.1609|[-0.0674,+0.4327]|
|R-V|high_margin|476|413|+0.0729|[-0.0793,+0.2695]|
|R-V|switch|1148|962|+0.1192|[-0.0621,+0.3571]|
|R-V|stable|60|44|+0.0957|[-0.0938,+0.3251]|
|R-V|gt_top3|995|797|+0.1348|[-0.0389,+0.3708]|
|R-V|gt_not_top3|213|209|+0.0154|[-0.2324,+0.2826]|
|R-A|low_margin|571|720|-0.1725|[-0.6414,+0.1477]|
|R-A|high_margin|372|427|-0.0637|[-0.2849,+0.0890]|
|R-A|switch|900|1094|-0.1243|[-0.4830,+0.1179]|
|R-A|stable|43|53|-0.0598|[-0.2431,+0.1119]|
|R-A|gt_top3|742|934|-0.1308|[-0.4333,+0.0729]|
|R-A|gt_not_top3|201|213|-0.0462|[-0.6156,+0.3338]|
Frozen first 300 manifest entries, prior to outcome inspection. Margin threshold 0.048815809190273285 is the pooled valid Base top1 minus top2 median. Switch is a change in Base runner-up identity at any valid 8-neighbor, not model-specific. GT rank is from Base raw scores. The 24x24 labels and predictions use nearest-neighbor mapping after mapper resize and padding; these are coarse cells, not all image pixels. Switch covers 156081/172800 cells (90.3%), so it is not a selective boundary-only set.

Full-pixel CIs use 2000 image bootstrap replicates seed 42 on pooled net correctness divided by valid pixels, not mIoU and not seed uncertainty. No principal V-M or R-V regional CI establishes reliable benefit. R-V full-pixel estimate is positive but CI spans zero. Positive and negative image counts are similar. Trained checkpoints also have different optimization trajectories; these are not isolated runtime ablations.

R-V corrects GT grass tree->grass 987996 and GT tree grass->tree 538914, but changes correct GT tree tree->grass 417387 and correct GT road road->grass 198585. V-M changes correct grass grass->tree 390369 and corrects tree grass->tree 373830. error_transitions.csv includes wrong-to-wrong transitions; paired_class_summary.csv and paired_class_per_image.csv preserve class effects.

Coverage: 34/35 GT classes present; windmill absent, train station appears in one image and dam in six. On the coarse grid they appear in one and four images. Rare-class bootstrap intervals are not reliable. No verified scene labels are available; scene representativeness is unknown. Manifest-order bias is possible; selection was not changed.

Prior 100-image bypass measures dependence of a co-trained system, not superiority to independent Base. Alternate rival holds the old margin fixed, creating inconsistent conditions; R also changes same_rival. Subset mIoU is not full 4700-image mIoU. Incorrect rival selection, spatial switching, host redundancy and coadaptation/optimization remain unproven explanations.
