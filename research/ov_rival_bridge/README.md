# Rival-conditioned spatial transfer: research prototype

**Status (2026-11-13):** New hypothesis and trainable implementation. Isolated CPU tests and a constructed-target optimizer/resume smoke passed locally; **no CUDA full-host parity, checkpoint/resume, segmentation training, multi-seed, or benchmark gain has been verified**. Do not claim novelty or SOTA.

## Motivation and relation to prior work

CAT-Seg (Cho et al., CVPR 2024) encodes image/text similarity as a cost volume and aggregates spatial and class evidence. PCA-Seg (Yin et al., CVPR 2026) challenges serial aggregation and offers parallel spatial/class aggregation with EPL/FOD. The existing OV Competition V1 uses local self-class scores (B) or relative cross-class *scores* (C) to condition the same 3x3 residual update. R here differs from C: it can read the intermediate feature **of the winning rival class**, not only the strongest rival score. It is not the PCA-Seg parallel aggregation or the original C formula.

Important caveats: rival features come from the host's first spatial Swin block and may already mix evidence; a top-score rival is not a ground-truth competing object or calibrated uncertainty; spatially changing the top rival can be noisy. The method might be redundant or harmful. No source code from third-party implementations is copied into this research module.

References:
- CAT-Seg: https://openaccess.thecvf.com/content/CVPR2024/html/Cho_CAT-Seg_Cost_Aggregation_for_Open-Vocabulary_Semantic_Segmentation_CVPR_2024_paper.html
- PCA-Seg: https://openaccess.thecvf.com/content/CVPR2026/html/Yin_PCA-Seg_Revisiting_Cost_Aggregation_for_Open-Vocabulary_Semantic_and_Part_Segmentation_CVPR_2026_paper.html
- OV Competition V1: `research/ov_competition_v1/METHOD_SPEC.md`

## Computational hypothesis

For each pixel i and candidate class c, define the winning rival r(i,c) = argmax_{k != c} s(i,k), using pre-aggregation similarity s. The model's first native spatial Swin stage yields h(i,c) in R^D. Define rival-relative features d(i,c)=LN(h(i,c))-LN(h(i,r(i,c))) and relative evidence m(i,c)=s(i,c)-s(i,r(i,c)).

R constructs 3x3 directional neighbor weights from a low-dimensional learned Q/K similarity plus a 7-dimensional pair descriptor: center/neighbor s, their difference, center/neighbor m, their absolute difference, and an indicator that the top rival identity is shared by center and neighbor. The model computes two **spatially local** difference messages, one in normalized class features h and one in rival-relative features d. It combines them with a pixel/class-dependent sigmoid mixing coefficient and a hard-sigmoid acceptance gate; a learnable small residual scale preserves the original h initially:

```text
h' = h + alpha * gate(s,margin) * W_out(
     (1-mix)*NeighborDelta(W_value LN(h))
       + mix*NeighborDelta(W_value [LN(h_c)-LN(h_r)]) )
```

Scores are not treated as confidence probabilities; both messages use the same neighborhood attention in R. No class ID embedding is used; the module should be category-permutation equivariant absent exact score ties. Invalid classes/positions are masked. With one class, the rival arms R and P are strict identity.

**Why this is not merely stacking blocks:** the research variable is the access to *rival-conditioned feature contrast* and its influence on what spatial messages are accepted, not an arbitrary extra convolution. Local 1x1 convolutions implement channel projections; attention implements selective 3x3 neighbor exchange; residuals limit perturbation of the native CAT-Seg representation. Separate ablations isolate these roles.

## Ablation arms

| Arm | Local neighbor transfer | Score margin | Rival feature contrast | Purpose |
| --- | --- | --- | --- | --- |
| Native | No additional module | No | No | Existing CAT-Seg A |
| S | Yes | No | No | Parameterized spatial-only control |
| M | Yes | Yes | No | Margins conditioned on spatial exchange |
| V | Yes (M attention retained) | Yes | Only rival Value message | Isolates rival-feature message contribution |
| P | No | Yes | Yes | Rival contrast without spatial mixing |
| R | Yes | Yes | Yes | Proposed joint interaction |

S/M/V/R instantiate identical trainable parameter names/shapes; some paths are inactive by design. P does not match FLOPs or active capacity and is a mechanism control, not a full compute-matched baseline. Existing Competition V1 A/B/C must remain separate training runs, **not combined** with R.

## Implementation and execution

Based on `feat/ov-competitive-spatial-update`; its core CAT-Seg and Competition V1 code is kept intact. This branch adds a new optional `rival_update` at the same location: **after the first native Swin and before the class transformer**, using aligned raw similarity from the host after any native top-K truncation. Default config leaves both optional methods off. Only serial CAT-Seg, single prompt, no cost-context/CAD are supported for R. The training initialization loads the same full CAT-Seg checkpoint **before** attaching the new module and building the optimizer/DDP. Checkpoint metadata records arm, rank, attention dimension, initialization scale, and base weights SHA-256. Rival evaluation now uses the same per-image statistics and iteration-specific output directories as Competition V1. These statistics do not encode pixelwise paired model-to-model correction maps. The existing train/evaluation entry `train_ovrs.py` retains the dataset manifest, seed, source-dev selection, original eval code, CLIP Q/V policy, batch/steps and provenance.

Run CPU tests and a six-step synthetic optimization/resume smoke experiment (these are **not** segmentation benchmarks), without needing Detectron2:

```bash
python -m pytest -q research/ov_rival_bridge/test_bridge.py research/ov_rival_bridge/test_mechanism_isolation.py research/ov_rival_bridge/test_install.py
python -m research.ov_rival_bridge.smoke_experiment
```

Use a separate working tree and fresh output directory for each arm. On a server with the historical CAT-Seg environment and a **trusted** common complete pretrained checkpoint:

```bash
python train_ovrs.py \
  --dataset-manifest /ABSOLUTE/PATH/manifest.json \
  --experiment baseline \
  --weights /ABSOLUTE/PATH/base_full_catseg.pth \
  --rival-arm R \
  --rival-base-weights /ABSOLUTE/PATH/base_full_catseg.pth \
  --batch-size 4 --max-iter 40000 \
  --output /ABSOLUTE/PATH/NEW_run_R \
  --num-gpus 1 \
  MODEL.OVRS.RIVAL_RANK 24 MODEL.OVRS.RIVAL_ATTN_DIM 8 \
  SEED 42
```

For S/M/V/P, change `--rival-arm` and use distinct fresh outputs. Before any long run, perform a true CUDA 2-update smoke test and full-weight bypass equivalence verification. Reject a run whose new parameters are missing from the optimizer or a resume that fails checkpoint metadata verification. If using `--eval-only`, retain the required common base checkpoint for model construction and supply the candidate checkpoint with `--weights`; do not conflate common initializer with the candidate to be evaluated.

**Note:** the existing train entry's eval path and checkpoint registration are reused rather than independently reimplemented. This branch has not yet been validated on the server or proven to load/restart a real new-arm checkpoint; use the smoke test as a gate.

## Scientific evaluation plan

1. Compare Native A, Competition V1 B/C, and S/M/V/P/R on the identical OVRSIS95K source train/dev protocol. Select checkpoints using **only source dev**; record exact step and version. The prior C vs A external DLRSD gain was <1 pp on one seed and is not an established improvement.
2. Inspect category-level change, class margins, gate/update statistics, per-image failure and correct-to-incorrect transitions. Specifically test whether rival feature contrast adds something beyond margin-conditioned spatial exchange (R vs M); test R vs P to determine whether spatial mixing is needed, while acknowledging unmatched capacity.
3. If there is credible consistent gain, repeat >=3 seeds and evaluate on a new, independent external benchmark. The historical DLRSD 200-image development set has already influenced OV research decisions and cannot be treated as untouched test.
4. Strong negative checks: R no better than M; improvement explained by larger update magnitude; added input class causing large output disturbance; performance concentrated in one class; ties/padding/truncation irregularities; unacceptable inference memory/latency.

Stop iterating this particular architecture if R fails to outperform simpler matched baselines after correct integration. This prototype is a scientific candidate, not a pre-selected paper contribution.
