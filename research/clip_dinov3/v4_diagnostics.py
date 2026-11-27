"""Opt-in, same-checkpoint diagnostics for the two matched V4 baselines."""

import json
import os
from pathlib import Path
import subprocess
import sys

from isaid_protocol import file_sha256, read_json


HERE = Path(__file__).resolve().parent
NATURAL_PROMPT = "a photo of a {}."
AERIAL_PROMPT = "an aerial image of a {}."
CASES = (
    ("baseline", (), (384,), (0,), (NATURAL_PROMPT,), 0.),
    *((f"resolution-{size}", ("--eval-sizes", str(size)), (size,), (0,), (NATURAL_PROMPT,), 0.)
      for size in (224, 512, 768)),
    *((f"rotate-{angle}", ("--eval-rotations", str(angle)), (384,), (angle,), (NATURAL_PROMPT,), 0.)
      for angle in (90, 180, 270)),
    ("rotate-ensemble", ("--eval-rotations", "0", "90", "180", "270"),
     (384,), (0, 90, 180, 270), (NATURAL_PROMPT,), 0.),
    ("scale-ensemble", ("--eval-sizes", "224", "384", "512"), (224, 384, 512), (0,), (NATURAL_PROMPT,), 0.),
    *((f"native-{weight:g}", ("--native-clip-weight", str(weight)), (384,), (0,), (NATURAL_PROMPT,), weight)
      for weight in (.25, .5, 1.)),
    ("aerial-prompt", ("--eval-prompt", AERIAL_PROMPT), (384,), (0,), (AERIAL_PROMPT,), 0.),
    ("prompt-ensemble", ("--eval-prompt", NATURAL_PROMPT, "--eval-prompt", AERIAL_PROMPT),
     (384,), (0,), (NATURAL_PROMPT, AERIAL_PROMPT), 0.),
)


def check_report(path, expected, flags):
    sizes, rotations, prompts, weight = flags
    report = read_json(path)
    if not (
        all(report.get(key) == value for key, value in expected.items())
        and report['inference_sizes'] == list(sizes)
        and report['inference_rotations'] == list(rotations)
        and report['inference_prompts'] == list(prompts)
        and report['native_clip_weight'] == weight
        and report['miou_percent'] is not None
    ):
        raise ValueError(f"Diagnostic report does not match its checkpoint or protocol: {path}")
    return report


def run_diagnostics(run, gpu, args):
    root = args.output / run.id
    checkpoint = root / "main" / run.variant / "train" / "final.pt"
    if not checkpoint.is_file():
        return {"state": "blocked", "reason": f"Missing final checkpoint: {checkpoint}"}
    audit = read_json(args.output / "isaid_data_audit.json")
    results = {}
    out = root / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    expected = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest": audit["source_manifest_sha256"],
        "data_sha256": audit["data_sha256"],
        "checkpoint_epoch": run.epochs,
        "eval_samples": 512, "eval_sample_seed": 42, "images": 512,
    }
    cmd = [sys.executable, "-u", HERE / "run_combinations.py", "eval",
           "--variant", run.variant, "--checkpoint", checkpoint,
           "--manifest", args.manifest, "--data-sha256", audit["data_sha256"],
           "--clip-weights", args.clip_weights, "--dino-weights", args.dino_weights,
           "--seed", str(run.seed), "--eval-samples", "512", "--eval-sample-seed", "42",
           "--batch-size", "1", "--warmup", "5", "--eval-loss", "--workers", "2"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    sample_hash = None
    for name, options, *flags in CASES:
        dest = out / name
        dest.mkdir(exist_ok=True)
        report_file = dest / "report.json"
        try:
            if not report_file.is_file():
                with (dest / "console.log").open("ab") as log:
                    subprocess.run([*cmd, *options, "--output", dest], cwd=HERE, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            report = check_report(report_file, expected, flags)
            if sample_hash is None:
                sample_hash = report["sample_ids_sha256"]
            if report["sample_ids_sha256"] != sample_hash:
                raise ValueError("Diagnostic sample IDs differ")
            results[name] = {"state": "complete", "miou_percent": report["miou_percent"],
                             "ms_per_image": report["ms_per_image"],
                             "report": str(report_file.relative_to(root))}
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            results[name] = {"state": "failed", "reason": str(exc)}
        temporary = out / "summary.tmp"
        temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(out / "summary.json")
    return {"state": "failed" if any(r["state"] == "failed" for r in results.values()) else "complete",
            "cases": results}
