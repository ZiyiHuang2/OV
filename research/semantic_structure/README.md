# Semantic–structure research prototype

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

Training cache must contain training images only. This small runner loads the cache into memory; it is intended for integration pilots, not full-scale 80k training. Random minibatches use cached views; they do not reproduce image augmentation. Encoder extraction, full dataset loader, official HG checkpoint migration, distributed training and dataset evaluation remain outstanding integration work. Never substitute random features for pretrained features in reported experiments.

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
