"""Matched evaluation jobs for existing 15-epoch iSAID checkpoints."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys

from isaid_protocol import file_sha256, read_json
from prepare_isaid import CLASSES, SPLIT_SIZES


HERE = Path(__file__).resolve().parent
VARIANTS = ("C0", "B0", "C1", "D0", "C2")
PROMPT = "a photo of a {}."
FOCAL_CLASSES = {
    "buildings": {"building", "buildings"},
    "vehicles": {"car", "cars", "vehicle", "vehicles", "large vehicle", "small vehicle"},
}
TRAIN_FIELDS = ("manifest", "weights", "config", "seed", "size", "batch_size", "lr",
                "weight_decay", "training_policy", "alignment_weight", "limit", "data_sha256")


@dataclass(frozen=True)
class HistoryRun:
    id: str
    variant: str
    manifest: Path
    manifest_sha256: str
    source: bool = False
    native_clip: bool = False
    size: int = 384
    epochs: int = 15
    seed: int = 42


def history_runs(args):
    """List the independent evaluation jobs without creating output files."""
    runs = []
    if args.manifest and args.manifest.is_file():
        digest = file_sha256(args.manifest)
        runs.extend(HistoryRun(job_id, "C0", args.manifest, digest, source=True, native_clip=native)
                    for job_id, native in (("HIST-iSAID-C0", False), ("HIST-iSAID-native-CLIP", True)))
    seen = set()
    for path in args.external_manifest:
        if not path.is_file():
            continue
        digest = file_sha256(path)
        if digest in seen:
            continue
        seen.add(digest)
        runs.extend(HistoryRun(f"HIST-{digest[:16]}-{variant}", variant, path, digest)
                    for variant in VARIANTS)
    return runs


def check_training(archive, run, source_sha256):
    training = read_json(archive / "train" / "report.json")
    steps = ((SPLIT_SIZES["train"] + 23) // 24) * run.epochs
    policy = {"epochs": run.epochs, "planned_steps": steps, "schedule": "cosine",
              "warmup_steps": 300, "min_lr_ratio": .1, "grad_clip": 1.}
    expected = {"schema": 2, "manifest": source_sha256, "train_classes": CLASSES,
                "size": run.size, "seed": run.seed, "steps": steps, "completed_epochs": run.epochs,
                "batch_size": 24, "training_policy": policy, "limit": None, "ignore_id": 255,
                "prompt": PROMPT, "split": "train", "dtype": "float32", "tf32": False,
                "lr": .001, "weight_decay": .01}
    if training["config"]["variant"] != run.variant or any(training.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Archived training report does not match the pinned iSAID E15 protocol: {archive}")
    return training


def class_errors(report):
    """Error counts from the ground-truth rows and prediction columns."""
    classes, matrix = report["test_classes"], report["confusion_matrix"]
    if (not classes or len(set(classes)) != len(classes) or len(matrix) != len(classes)
            or any(len(row) != len(classes) for row in matrix)
            or any(type(count) is not int or count < 0 for row in matrix for count in row)):
        raise ValueError("Confusion matrix does not match its vocabulary")
    result = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        row, col = matrix[i], [row[i] for row in matrix]
        ground_truth, predicted = sum(row), sum(col)
        union = ground_truth + predicted - tp
        result[name] = {
            "gt_pixels": ground_truth, "pred_pixels": predicted, "true_positive_pixels": tp,
            "false_negative_pixels": ground_truth - tp, "false_positive_pixels": predicted - tp,
            "recall": tp / ground_truth if ground_truth else None,
            "precision": tp / predicted if predicted else None,
            "iou": tp / union if union else None,
            "missed_as": {other: row[j] for j, other in enumerate(classes) if j != i and row[j]},
            "false_positive_from": {other: col[j] for j, other in enumerate(classes) if j != i and col[j]},
        }
    return {"unit": "pixels with valid evaluation labels", "class_errors": result,
            "focal_classes": {group: [name for name in classes if name.casefold() in aliases]
                              for group, aliases in FOCAL_CLASSES.items()}}


def expected_evaluation(run, training, checkpoint):
    manifest = read_json(run.manifest)
    classes = read_json(run.manifest.parent / manifest["test_classes"])
    samples = [row["sample_id"] for row in manifest["val"]]
    if not classes or len(set(classes)) != len(classes) or not samples or len(set(samples)) != len(samples):
        raise ValueError(f"Evaluation requires a unique, nonempty vocabulary and sample list: {run.manifest}")
    return {
        "manifest": run.manifest_sha256, "checkpoint_train_manifest": training["manifest"],
        "checkpoint_sha256": file_sha256(checkpoint), "checkpoint_kind": "evaluation_only",
        "checkpoint_epoch": run.epochs, "checkpoint_steps": training["steps"],
        "config": training["config"], "weights": training["weights"], "train_classes": CLASSES,
        "split": "val", "ignore_id": manifest["ignore_label"], "test_classes": classes,
        "images": len(samples), "sample_ids_sha256": sha256("".join(name + "\n" for name in samples).encode()).hexdigest(),
        "size": run.size, "seed": run.seed, "batch_size": 1, "dtype": "float32", "tf32": False,
        "prompt": PROMPT, "limit": None, "eval_samples": None, "force_keep_all": False,
        "inference_sizes": [run.size], "inference_rotations": [0], "inference_prompts": [PROMPT],
        "native_clip_weight": float(run.native_clip),
    }


def audit_split(run):
    if run.source:
        return "iSAID_split/val"
    audit_path = run.manifest.with_name("data_audit.json")
    if audit_path.is_file():
        try:
            audit = read_json(audit_path)
        except (OSError, ValueError):
            return None
        if isinstance(audit, dict):
            if audit.get("manifest_sha256") == run.manifest_sha256:
                return audit.get("published_split") or audit.get("original_loveda_split")
            if audit.get("val_manifest_sha256") == run.manifest_sha256:
                return audit.get("split")
    return None


def run_history(run, gpu, args):
    checkpoint = args.reference_root / run.variant / "train" / "final.pt"
    if not checkpoint.is_file():
        return {"state": "blocked", "reason": f"Missing historical checkpoint: {checkpoint}"}
    try:
        if file_sha256(run.manifest) != run.manifest_sha256:
            raise ValueError(f"Evaluation manifest changed after queue initialization: {run.manifest}")
        audit = read_json(args.output / "isaid_data_audit.json")
        if run.source and audit["source_manifest_sha256"] != run.manifest_sha256:
            raise ValueError("Historical source evaluation must use the audited iSAID manifest")
        training = check_training(args.reference_root / run.variant, run, audit["source_manifest_sha256"])
        expected = expected_evaluation(run, training, checkpoint)
        output = args.output / "history"
        if run.source:
            output = output / "isaid_val" / ("native-CLIP" if run.native_clip else run.variant)
            expected["data_sha256"] = audit["data_sha256"]
        else:
            output = output / "external" / run.manifest_sha256[:16] / run.variant / "eval"
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "report.json"
        if not report_path.is_file():
            command = [sys.executable, "-u", HERE / "run_combinations.py", "eval",
                       "--variant", run.variant, "--checkpoint", checkpoint, "--manifest", run.manifest,
                       "--clip-weights", args.clip_weights, "--seed", str(run.seed), "--batch-size", "1",
                       "--warmup", "30", "--eval-loss", "--workers", "2", "--output", output]
            if run.variant != "C0":
                command.extend(("--dino-weights", args.dino_weights))
            if run.source:
                command.extend(("--data-sha256", audit["data_sha256"]))
            if run.native_clip:
                command.extend(("--native-clip-weight", "1"))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
            with (output / "console.log").open("ab") as log:
                subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read_json(report_path)
        if (any(report[key] != value for key, value in expected.items()) or not isfinite(report["miou_percent"])
                or any(report["checkpoint_training"].get(key) != training.get(key) for key in TRAIN_FIELDS)):
            raise ValueError(f"Historical evaluation report does not match its checkpoint or protocol: {report_path}")
        errors = class_errors(report)
        split = audit_split(run)
        errors.update(report=str(report_path.relative_to(args.output)), checkpoint_sha256=report["checkpoint_sha256"],
                      manifest=report["manifest"], manifest_path=str(run.manifest),
                      published_split=split, sample_ids_sha256=report["sample_ids_sha256"])
        temporary = output / "class_errors.tmp"
        temporary.write_text(json.dumps(errors, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "class_errors.json")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        return {"state": "failed", "reason": str(exc)}
    return {"state": "complete", "manifest": str(run.manifest), "published_split": split, "images": report["images"],
            "checkpoint_sha256": report["checkpoint_sha256"], "miou_percent": report["miou_percent"],
            "report": str(report_path.relative_to(args.output))}
