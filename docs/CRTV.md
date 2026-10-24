# CRTV pilot branch

This branch adds a small, opt-in **Counterfactual Relation-aware Text
Verifier (CRTV)** to the CAT-Seg cost path.  It is an experimental diagnostic,
not a claim that the full counterfactual training method is complete.

The verifier compares each class response with a text-relation-weighted set of
competing class responses.  It adds a bounded residual to the existing
`[B, prompts, classes, H, W]` correlation tensor and keeps the class dimension
dynamic.  The baseline path is unchanged unless `MODEL.OVRS.CRTV.ENABLED`
is set to `true`.

## What this branch tests

The first test is whether class competition already explains a measurable
fraction of the errors before adding a new visual encoder or a spatial
refinement branch.  The pilot should therefore compare the baseline,
`crtv.yaml`, and an explicit local-cost refinement control under the same
checkpoint and vocabulary.

`RESIDUAL_SCALE: 0.0` is intentional.  It makes the enabled module an
identity at initialization while exposing a learnable residual scale for a
controlled training pilot.  A non-zero scale is an inference diagnostic only;
it must not be presented as a trained result.

## Current implementation boundary

Implemented:

- dynamic-vocabulary text relation weights with the diagonal removed;
- relation-weighted competing evidence and a bounded margin gate;
- identity-safe residual injection;
- diagnostics for positive evidence, negative evidence, margin, gate, and
  relation weights;
- configuration and checkpoint-visible opt-in wiring.

Intentionally deferred:

- scene-context swapping and counterfactual data construction;
- region-slot discovery;
- new losses or changes to the CLIP fine-tuning policy;
- full end-to-end claims about unseen-class generalization.

Those pieces should only be added after the frozen-checkpoint evidence test
shows that the verifier changes the expected failure group.

## Suggested pilot order

1. Run the existing baseline evaluation and save the per-class response and
   final prediction artifacts.
2. Run the enabled verifier with a non-zero scale as an offline diagnostic,
   keeping the model weights fixed.
3. Measure unseen mIoU, hIoU, target-class recall, correct-to-wrong and
   wrong-to-correct transfers, and high-confidence ranking flips.
4. Only if the diagnostic improves the intended failure group, train the
   verifier with CLIP frozen and compare against the baseline and the existing
   `full_interaction` prototype.

Do not interpret an overall mIoU increase alone as evidence that CRTV improves
open-vocabulary behavior.

For a tensor-only first pass, use `tools/crtv_tensor_diagnostic.py`.  It reads
cached correlation and text-feature tensors and writes the verified scores plus
detached diagnostics without importing Detectron2 or changing a checkpoint.
