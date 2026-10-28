# Semantic–structure research prototype

Historical v1 delivery documentation; subsequent real-data engineering results are in [EXECUTION_20261028.md](EXECUTION_20261028.md). The original implementation and this README are retained for reproducibility.

Status: feature-level experimental decoder, NOT an HG reproduction or a validated segmentation method.
No trained weights or measured dataset gains are included. Existing CAT-Seg training remains unchanged.

## Design review
Local description selection can reinforce false matches. Structural similarity need not coincide with semantic identity. Convex neighborhood transport cannot invent absent semantics; the following learned refinement is not itself constrained to convex outputs. Frozen encoders do not guarantee preserved unseen performance.
This first implementation therefore has a 2x2 factorial: mean/adaptive semantics x fusion/transport geometry. Same frozen feature inputs and labels must be used across arms. The original HG model is an external fifth baseline, not implemented here.

The prototype retains channel-wise image/text products, uses shared parameters for every class, masks padded descriptions and invalid spatial neighbors, and reinjects direct evidence at each decoder stage. It deliberately uses a simple cosine-based local selector rather than an additional learned expert router. Exact duplicate descriptions must be removed BEFORE encoding. Ontology relations are not implemented; do not call this a knowledge-graph reasoning model.

## Files and execution
Requires Python 3.10+ and PyTorch 2.2+.
Run from this directory:
```bash
python -m unittest -v test_model
python train_cached.py --cache /path/train_features.pt --output /path/run.pt --semantic mean --spatial fusion
```
Repeat with mean/transport, adaptive/fusion, adaptive/transport. Use identical seeds and update budgets.

Cache is a trusted torch weights-only dictionary:
- visual: float B,D,H,W, final spatial CLIP features in the TEXT-ALIGNED embedding space, not arbitrary intermediate tokens.
- text: float K,P,D from the SAME frozen VLM; descriptions, templates, and vocabulary order fixed and documented.
- valid: bool K,P, at least one description per class.
- structures: list of float B,C_l,H_l,W_l from one verified frozen structure encoder, coarse to fine.
- labels: long B,Hout,Wout, contiguous IDs 0..K-1; ignored pixels 255.
- metadata: plain dictionary recording image IDs, split, encoder checkpoint SHA256, preprocessing, class names, descriptions and provenance.

Training cache must contain training images only. This small runner loads the cache into memory; it is intended for integration pilots, not full-scale 80k training. Random minibatches use cached views; they do not reproduce image augmentation. A sampled all-ignore batch is skipped without an optimizer step; `--steps` counts sampled batches, not necessarily optimizer updates. Encoder extraction, full dataset loader, official HG checkpoint migration, distributed training and dataset evaluation remain outstanding integration work. Never substitute random features for pretrained features in reported experiments.

## Research gates
1. Verify real CLIP alignment and structure feature normalization, checkpoint identity and spatial correspondence.
2. Reproduce the external HG baseline on its documented source/target protocol.
3. Run four matched arms on source train with source validation selection; keep target test sealed.
4. Report target per-class IoU, precision/recall, seen/related/unseen semantics, runtime and peak memory. Confirm surviving comparisons across seeds.
5. Only after transfer is established, run downstream DLRSD seen adaptation.
If adaptive/transport does not beat mean/fusion, do not expand modules. No unseen-GT threshold or prompt tuning.

## Sources
Original implementation inspired by:
- KG-OVRSeg, DOI 10.1016/j.isprsjprs.2025.11.029 (semantic organization; this prototype does not implement its graph).
- HG-RSOVSSeg, DOI 10.3390/rs18020213 (channel-wise interaction and hierarchical class-conditioned decoding).
- RS-OVSGSeg, DOI 10.1109/TGRS.2026.3726176 (separate pretrained structure guidance).
No upstream code was copied. This is a new experimental implementation, not their official combined model.


## Verified real-feature engineering pilot

See [EXECUTION_20261028.md](EXECUTION_20261028.md) for the completed four-arm
DLRSD result and its limitations. It does not establish cross-dataset or unseen
performance. The registered experiment protocol, including exact sample indices,
is [protocols/dlrsd_pilot_20261028.json](protocols/dlrsd_pilot_20261028.json).

The new real-feature tools were verified with Python 3.10.14, PyTorch 2.5.1+cu121,
torchvision 0.20.1, open-clip-torch 2.16.0, timm 0.9.16, numpy 1.26.4 and Pillow
10.4.0. They use existing local weights and never request pretrained downloads.
The RSIB source is an explicit external input (RSKT-Seg commit
`7b84091598e1edc3236dfbf45cc27e7e3436ffcb`, `RSKT_Seg/vision_transformer.py`);
its hash and all encoder weight hashes are recorded in the cache metadata.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
python -m unittest -v test_model test_gpu
python extract_dlrsd.py --manifest /path/train_val/manifest.json   --protocol protocols/dlrsd_pilot_20261028.json   --clip /path/ViT-B-16.pt --rsib /path/RSIB.pth   --rsib-source /path/RSKT_Seg/vision_transformer.py --output /new/features
python run_pilot.py --features /new/features   --protocol protocols/dlrsd_pilot_20261028.json --output /new/engineering --phase engineering
# Only proceed after all four engineering checks pass.
python run_pilot.py --features /new/features   --protocol protocols/dlrsd_pilot_20261028.json --output /new/pilot --phase pilot
python check_resume.py --features /new/features --checkpoints /new/pilot   --output /new/resume_check.json
```

Extraction uses a common 224x224 full-image resize, projected final CLIP patch
tokens and final RSIB patch features. Patch projection is an extrapolation of
CLIP's global alignment, not proof of strong dense semantics. Descriptions are
exactly deduplicated before encoding. Encoders are frozen and their in-memory
state hashes must remain unchanged. Train/val are taken only from the explicitly
fixed manifest entries. Val original IDs map to the 14 seen classes; unseen
pixels are ignored. All selected images, masks, original manifest splits,
class order, descriptions, validity masks and encoder identities accompany the
cache. A source custom split is not an official upstream split.

`run_pilot.py` applies the same initialization and independent sampling generator
to all arms, logs sampled indices/loss/time, saves model/optimizer/sampler state,
and evaluates only the fixed endpoint. `check_resume.py` restores that state and
verifies exact next-update model and optimizer equality; it does not add its
check update to reported experimental results. The pilot runner counts actual
optimizer updates; the original lightweight `train_cached.py` still counts sampled
batches. No best-validation selection or unseen calibration is performed.

Metrics include raw total/per-image confusion matrices, per-class IoU, precision,
recall, and mIoU over classes with nonzero union. Absent values are null. Padded
text values must remain finite. Effective parameter counts exclude mode-inactive
scalars and the shared classifier bias that cancels in class softmax; no parameters
are added to equalize arm sizes. The current mean/adaptive comparison also includes
the difference in descriptor normalization, so it is not a pure selector-only
causal comparison.
