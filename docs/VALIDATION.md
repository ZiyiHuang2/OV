# Pre-release validation

Checks performed on 2026-10-10 with Python 3.12, PyTorch 2.5.1+cpu,
torchvision 0.20.1+cpu, timm 0.9.16 and the pinned Detectron2 revision in
UPSTREAM.json. Detectron2 CPU extensions were built locally with gcc/g++.

## Passed checks

- `python -m pytest -q`: 50 tests passed. Coverage includes module forward
  and backward, finite and nonzero new-module gradients, dynamic class counts
  (1, 3 and 17), class permutation, CAD isolation between images, masked BCE,
  text vocabulary/cache/gradient behavior, label conversion and all configs.
- `python tools/check_baseline_equivalence.py`: the disabled-modules
  Aggregator matches the pinned upstream parameter keys and output exactly
  (`max_error=0.0`). This comparison does not cover the changed training loss.
- `python tools/smoke_aggregator.py --classes 17 --full`: CPU forward/backward
  passed for the 24×24 aggregation grid and output `[1,17,96,96]`; the
  aggregation/decoder contains 3,079,699 parameters. This excludes CLIP and
  does not measure complete GPU training memory.
- Complete `train_ovrs.py` pipeline: pretrained CLIP download, reading six
  synthetic training images, three iterations with all three modules, finite
  loss, checkpoint saving and evaluation on two synthetic validation images.
  Training used two seen classes and evaluation used three total classes,
  exercising the changing-vocabulary path and seen/unseen reporting.
- `--eval-only --weights .../model_final.pth`: loading the saved full-model
  checkpoint and evaluating the synthetic validation split succeeded.
- Full-model `--resume` restored the saved optimizer/iteration state and
  continued from iteration 3 to iteration 4.
- Strict evaluation rejects missing, unexpected or incorrectly shaped
  checkpoint weights; module wiring hooks confirm LSK runs once per image
  batch, both parallel branches read the same cost once, and CAD sees real
  classes after internal padding is removed. Zero-scale tests check that
  fusion retains the branch mean and CAD retains incoming guidance.
- Python compilation, shell syntax and `git diff --check` passed.

The integration run used `--num-gpus 0 MODEL.DEVICE cpu
SOLVER.AMP.ENABLED False DATALOADER.NUM_WORKERS 0`, with batch size 1.
Synthetic examples validate the pipeline, not segmentation quality.

## Not validated

- No NVIDIA GPU was available: CUDA installation, AMP training and 4090
  peak memory were not tested locally.
- No real DLRSD split, label map, external dataset or user checkpoint was
  supplied. No benchmark gain or superiority of the module combination is
  claimed. This is an ablation-ready implementation.
- GitHub Actions is provided separately; local passes do not imply that
  the remote CI run has finished.

All experiments share the corrected ignore-pixel BCE. Consequently the
baseline is structurally identical when modules are disabled, but its
training objective is not an exact reproduction of upstream ignored-pixel
handling. EPL uses GroupNorm and is explicitly an adaptation.

## Selective design validation

`full_selective` adds branch gating and weighted CAD while retaining the
previous `full` computation. The full-size selective aggregator passed
forward/backward with `[1,17,96,96]` output and 2,476,271 parameters.
Tests additionally cover selectable branches, uniform/scored class pooling,
class permutation with padding, whole-vocabulary duplication and CPU
bfloat16 autocast backward. CPU autocast is not CUDA AMP validation.

The complete pretrained-CLIP selective pipeline was trained for three CPU
iterations on the same six synthetic images (two train classes), followed
by evaluation on two images with three test classes. The saved selective
checkpoint was reloaded for evaluation. The previous full-model checkpoint
was also loaded strictly and evaluated successfully. Synthetic accuracy
is not reported as evidence of improved segmentation.

## Repository organization validation

After separating the three modules and moving the shared Trainer into the
package, all 50 tests passed again. All 13 experiment configurations resolve
the relocated base YAMLs. The main and archived Trainer CLI help commands,
Python compilation, shell syntax and the active documentation links passed.

With the same seed, the baseline, `full` and `full_selective` aggregators have
exactly the same parameter keys, initialized tensors and forward outputs as
the pre-organization code (`max_output_error=0.0`). Both full-sized 17-class
aggregators passed CPU forward/backward again with unchanged parameter counts.
The previously saved complete selective CLIP checkpoint was loaded strictly
and evaluated on the two synthetic validation images through the reorganized
entry point. No architecture or training-objective changes were made in this
organization pass. Archived demos and natural-image preparation scripts were
compiled, but were not exercised with their original data/dependencies.
