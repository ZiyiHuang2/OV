"""Resolve and run the matched iSAID experiment queue through the formal launcher."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys

from isaid_protocol import audit_isaid, file_sha256
from prepare_isaid import CLASSES, SPLIT_SIZES


HERE = Path(__file__).resolve().parent
FORMAL_PROMPT = "a photo of a {}."


@dataclass(frozen=True)
class Run:
    id: str
    variant: str
    epochs: int = 15
    split: str = ""
    seed: int = 42
    control: str = ""
    size: int = 384


def held_pair(split, epochs=15, seed=42):
    suffix = "" if epochs == 15 and seed == 42 else f"-e{epochs}-s{seed}"
    return [Run(f"{split}{i}{suffix}", variant, epochs, split, seed)
            for i, variant in enumerate(("C0", "B0"))]


FIXED = (
    # P0: held-class evidence.
    *held_pair("H1"), *held_pair("H2"), *held_pair("H1", 40),
    # P1: visual, spatial, and early/late fusion controls; long-run anchors.
    *(Run(v, v) for v in ("D01", "D02", "E01", "E02", "F01", "F02")),
    *(Run(f"T0{i}", v, 40) for i, v in ((0, "C0"), (1, "B0"), (3, "C1"))),
    # P2: detailed fusion and supervision controls.
    *(Run(f"F{i:02d}", "C2", control=c) for i, c in enumerate(
        ("mean", "learned", "separate", "noalign", "classpool"), 3)),
    Run("F08", "F08"), Run("F09", "F09"),
    *(Run(f"T0{i}", v, 40) for i, v in ((2, "D0"), (4, "C2"))),
    *(Run(f"{v}-H1", v, split="H1") for v in ("D0", "C1", "C2")),
    # P3: second long-run split and seed replication.
    *held_pair("H2", 40), *held_pair("H1", seed=43), *held_pair("H2", seed=43),
)

RESERVE = (
    *(Run(f"{v}-s43", v, seed=43) for v in ("C0", "B0", "D0", "C1", "C2")),
    *(Run(f"{v}-H2", v, split="H2") for v in ("D0", "C1", "C2")),
    *(Run(f"{v}-H1", v, split="H1") for v in ("D01", "D02", "F01", "F02")),
)

# The 512 pair is an input-size diagnostic, with the matching 384 controls
# T01/T03. The relation pair shares its 40-epoch D0/C2 anchors T02/T04.
EXTRA = {
    "resolution": (Run("B0-512", "B0", 40, size=512), Run("C1-512", "C1", 40, size=512)),
    "relations": (Run("C3-e40", "C3", 40), Run("C4-e40", "C4", 40)),
}
EXTRA_ANCHORS = {"resolution": ("T01", "T03"), "relations": ("T02", "T04")}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def dev_result(path, run, *, development=True):
    data = read_json(path)
    config = data["config"]
    if (config["variant"] != run.variant or data["checkpoint_epoch"] != run.epochs
            or data["checkpoint_kind"] != "evaluation_only"):
        raise ValueError(f"Unexpected variant or checkpoint: {path}")
    # The archived E15 evaluator's top-level batch and optimizer arguments
    # describe inference. Read the sibling training report for those runs.
    training = data.get("checkpoint_training")
    if training is None:
        training = read_json(path.parents[1] / "train" / "report.json")
        if training["completed_epochs"] != run.epochs or training["steps"] != data["checkpoint_steps"]:
            raise ValueError(f"Unexpected legacy training completion: {path}")
    elif not (len(data["checkpoint_sha256"]) == 64 and all(c in "0123456789abcdef" for c in data["checkpoint_sha256"])):
        raise ValueError(f"Invalid checkpoint digest: {path}")
    for field in ("weights", "config", "size", "seed", "training_policy", "alignment_weight"):
        if training[field] != data[field]:
            raise ValueError(f"Evaluation and training {field} differ: {path}")
    policy = training["training_policy"]
    batch_size = training["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError(f"Expected a positive training batch: {path}")
    updates = math.ceil(SPLIT_SIZES["train"] / batch_size)
    settings = (policy["schedule"], policy["warmup_steps"], policy["min_lr_ratio"], policy["grad_clip"],
                training["lr"], training["weight_decay"])
    expected = ("cosine", min(300, updates * run.epochs // 10), .1, 1., .001, .01)
    if (run.split or (training["seed"], training["size"]) != (run.seed, run.size)
            or training["manifest"] != data["checkpoint_train_manifest"]
            or data["manifest"] != data["checkpoint_train_manifest"]
            or training["limit"] is not None or data["limit"] is not None
            or training.get("data_sha256") != data.get("data_sha256")
            or policy["epochs"] != run.epochs
            or policy["planned_steps"] != updates * run.epochs
            or settings != expected
            or data["checkpoint_steps"] != policy["planned_steps"]
            or data["split"] != "val" or data["train_classes"] != CLASSES or data["test_classes"] != CLASSES
            or (data["batch_size"], data["ignore_id"], data["dtype"], data["tf32"]) != (1, 255, "float32", False)
            or data["force_keep_all"]
            or data.get("inference_sizes", [run.size]) != [run.size]
            or data.get("inference_rotations", [0]) != [0]
            or data["prompt"] != FORMAL_PROMPT
            or data.get("inference_prompts", [FORMAL_PROMPT]) != [FORMAL_PROMPT]
            or data.get("native_clip_weight", 0) != 0):
        raise ValueError(f"Unexpected training or evaluation protocol: {path}")
    if development:
        if (data["images"], data["eval_samples"], data["eval_sample_seed"]) != (512, 512, 42):
            raise ValueError(f"Expected the fixed development subset: {path}")
    elif data["images"] != SPLIT_SIZES["val"] or data["eval_samples"] is not None:
        raise ValueError(f"Expected the complete validation split: {path}")
    if run.variant == "C2":
        fusion = run.control if run.control in ("mean", "learned", "separate") else "context"
        region = "class" if run.control == "classpool" else "component"
        alignment = 0 if run.control == "noalign" else .2
        if (config["fusion"], config["region_mode"], data["alignment_weight"]) != (fusion, region, alignment):
            raise ValueError(f"Unexpected C2 control: {path}")
    if run.variant == "F08" and (data.get("region_to_text_loss_fraction"), data["alignment_weight"],
                                  training.get("region_to_text_loss_fraction")) != (.5, .2, .5):
        raise ValueError(f"Expected matched F08 region-to-text weight: {path}")
    matrix = data["confusion_matrix"]
    classes = len(data["test_classes"])
    if (len(matrix) != classes or any(len(row) != classes for row in matrix)
            or any(type(count) is not int or count < 0 for row in matrix for count in row)):
        raise ValueError(f"Confusion matrix does not match the vocabulary: {path}")
    populations = [sum(row) for row in matrix]
    scores = [100 * matrix[i][i] / (support + sum(row[i] for row in matrix) - matrix[i][i])
              for i, support in enumerate(populations) if support]
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError(f"No valid supported-class score: {path}")
    identity = {key: data[key] for key in ("manifest", "weights", "test_classes", "size", "seed",
                                           "ignore_id", "prompt", "dtype", "tf32")}
    identity.update({key: training[key] for key in ("batch_size", "lr", "weight_decay")})
    identity["schedule"] = {key: value for key, value in policy.items() if key not in ("epochs", "planned_steps")}
    subset = data["sample_ids_sha256"], populations
    return identity, subset, training.get("data_sha256"), sum(scores) / len(scores)


def promotions(root, reference_root, minimum_delta, limit, requested_ids=None, audit=None):
    fixed = {run.id: run for run in FIXED}
    requested = set(requested_ids) if requested_ids is not None else None
    long_anchor = {"C0": "T00", "B0": "T01", "C2": "T04"}
    groups = [(("D01",), "B0"), (("D02",), "C0"), (("E01", "E02"), "B0"),
              (("F01", "F02"), "B0")]
    groups += [((f"F{i:02d}",), "C2") for i in range(3, 10)]
    eligible, decisions = [], []
    for ids, anchor in groups:
        if requested is not None and not set(ids) <= requested:
            continue
        reference = reference_root / anchor / "dev_epoch_015" / "report.json"
        rows = [root / name / "main" / fixed[name].variant / "dev_epoch_015" / "report.json" for name in ids]
        decision = {"ids": ids, "anchor": anchor}
        if not reference.is_file() or not all(row.is_file() for row in rows):
            decisions.append({**decision, "state": "missing_reports"})
            continue
        try:
            anchor_key, anchor_subset, anchor_data, anchor_score = dev_result(reference, Run(anchor, anchor))
            results = [dev_result(row, fixed[name]) for name, row in zip(ids, rows, strict=True)]
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            decisions.append({**decision, "state": "invalid_report", "reason": str(exc)})
            continue
        audited_data = results[0][2]
        if (not audited_data or (anchor_data is not None and anchor_data != audited_data)
                or (audit is not None and (anchor_key["manifest"], audited_data)
                    != (audit["source_manifest_sha256"], audit["data_sha256"]))
                or any((key, subset, content) != (anchor_key, anchor_subset, audited_data)
                       for key, subset, content, _ in results)):
            decisions.append({**decision, "state": "protocol_mismatch"})
            continue
        required = [(long_anchor[anchor], anchor)]
        if ids == ("E01", "E02"):
            required.append(("T03", "C1"))
        long_reports = [root / run_id / "main" / variant / "eval" / "report.json"
                        for run_id, variant in required]
        if not all(path.is_file() for path in long_reports):
            decisions.append({**decision, "state": "missing_long_anchor"})
            continue
        try:
            anchors = [dev_result(path, fixed[run_id], development=False)
                       for path, (run_id, _) in zip(long_reports, required, strict=True)]
            valid = all(key == anchor_key and content == audited_data for key, _, content, _ in anchors)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            valid = False
        if not valid:
            decisions.append({**decision, "state": "invalid_long_anchor"})
            continue
        delta = max(score for _, _, _, score in results) - anchor_score
        decision["delta_pp"] = delta
        if delta >= minimum_delta:
            eligible.append((delta, ids, decision))
        else:
            decisions.append({**decision, "state": "below_threshold"})
    selected = []
    for _, ids, decision in sorted(eligible, key=lambda result: -result[0]):
        if len(selected) + len(ids) > limit:
            decisions.append({**decision, "state": "budget"})
            continue
        selected.extend(replace(fixed[name], id=f"{name}-e40", epochs=40) for name in ids)
        decisions.append({**decision, "state": "selected"})
    return selected, decisions


def extra_jobs(runs, args, audit):
    fixed = {run.id: run for run in FIXED}
    enabled, decisions = [], {}
    if not any(job in runs for jobs in EXTRA.values() for job in jobs):
        return enabled, decisions
    weights = {"clip": file_sha256(args.clip_weights), "dinov3": file_sha256(args.dino_weights)}
    for group, jobs in EXTRA.items():
        wanted = [job for job in jobs if job in runs]
        if not wanted:
            continue
        missing = []
        for job_id in EXTRA_ANCHORS[group]:
            anchor = fixed[job_id]
            report = args.output / job_id / "main" / anchor.variant / "eval" / "report.json"
            try:
                key, _, content, _ = dev_result(report, anchor, development=False)
                if (key["manifest"] != audit["source_manifest_sha256"] or key["batch_size"] != args.batch_size
                        or key["weights"] != weights
                        or content != audit["data_sha256"]):
                    raise ValueError("dataset, encoder weights or training batch differs")
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                missing.append({"anchor": job_id, "reason": str(exc)})
        if missing:
            decisions.update((job.id, {"state": "blocked", "anchors": missing}) for job in wanted)
        else:
            enabled.extend(wanted)
    return enabled, decisions


def resolve_gpus(gpus):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True)
    inventory = {}
    for line in result.stdout.splitlines():
        index, uuid = (value.strip() for value in line.split(","))
        inventory[index] = inventory[uuid] = uuid
    try:
        resolved = [inventory[gpu] for gpu in gpus]
    except KeyError as exc:
        raise ValueError(f"Unknown physical GPU {exc.args[0]}") from exc
    if len(set(resolved)) != len(resolved):
        raise ValueError("--gpu must name distinct physical GPUs")
    return resolved


def run_job(run, gpu, args):
    out = args.output / run.id
    out.mkdir(exist_ok=True)
    spec = asdict(run)
    spec_path = out / "queue_spec.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError(f"Existing queue job has a different specification: {out}")
    write_json(spec_path, spec)
    env = {**os.environ, "PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": gpu,
           "CSERIES_GPU_IDS": "", "CSERIES_VARIANTS": run.variant,
           "CSERIES_EPOCHS": str(run.epochs), "CSERIES_HELD_SPLIT": run.split,
           "CSERIES_SEED": str(run.seed), "CSERIES_C2_CONTROL": run.control,
           "CSERIES_SIZE": str(run.size), "CSERIES_UNATTENDED": "1",
           "CSERIES_DATA_AUDIT": str(args.output / "isaid_data_audit.json")}
    launcher = ["bash", str(HERE / "run_isaid_suite.sh")]
    operands = [str(p) for p in (args.manifest, args.clip_weights, args.dino_weights, out)] + [str(args.batch_size)]
    stages = [("preflight", None), ("main", None)]
    if not run.split:
        stages += [("external", path) for path in args.external_manifest]
    failures = []
    with (out / "queue.log").open("a", encoding="utf-8") as log:
        for phase, external in stages:
            phase_env = {**env, "EXTERNAL_MANIFEST": str(external or "")}
            result = subprocess.run([*launcher, phase, *operands], cwd=HERE,
                                    env=phase_env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append({"phase": phase, "manifest": str(external) if external else None,
                                 "returncode": result.returncode})
                if phase != "external":
                    break
    return {"state": "failed", "failures": failures} if failures else {"state": "complete"}


def run_queue(runs, args):
    pending = Queue()
    for run in runs:
        pending.put(run)

    def consume(gpu):
        results = {}
        while True:
            try:
                run = pending.get_nowait()
            except Empty:
                return results
            try:
                status = run_job(run, gpu, args)
            except (OSError, ValueError) as exc:
                status = {"state": "failed", "reason": str(exc)}
            directory = args.output / run.id
            if directory.is_dir():
                try:
                    write_json(directory / "queue_status.json", status)
                except OSError as exc:
                    status = {"state": "failed", "reason": f"Cannot write queue status: {exc}",
                              "job_status": status}
            results[run.id] = status
            print(f"{run.id} {status['state']} on {gpu}", flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpu)) as pool:
        return {k: v for result in pool.map(consume, args.gpu) for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "clip-weights", "dino-weights", "output", "reference-root"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve())
    parser.add_argument("--external-manifest", action="append", type=lambda p: Path(p).resolve(), default=[])
    parser.add_argument("--gpu", action="append", help="Repeat for each GPU UUID or index")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--include-reserve", action="store_true")
    parser.add_argument("--extra", choices=EXTRA, action="append", default=[])
    parser.add_argument("--ids", nargs="+", help="Restrict to these exact job IDs")
    parser.add_argument("--promote-min-delta", type=float, help="Enable matched 40-epoch follow-ups at this threshold in pp")
    parser.add_argument("--promotion-limit", type=int, default=11)
    parser.add_argument("--execute", action="store_true", help="Start jobs, otherwise print the resolved plan")
    args = parser.parse_args()
    runs = [*FIXED, *(RESERVE if args.include_reserve else ())]
    runs.extend(run for name in dict.fromkeys(args.extra) for run in EXTRA[name])
    if args.ids:
        if len(set(args.ids)) != len(args.ids) or set(args.ids) - {r.id for r in runs}:
            parser.error("--ids must list distinct IDs from the enabled queue")
        runs = [run for run in runs if run.id in args.ids]
    if args.promote_min_delta is not None:
        if (args.reference_root is None or args.output is None or not math.isfinite(args.promote_min_delta)
                or args.promote_min_delta <= 0 or not 1 <= args.promotion_limit <= 11):
            parser.error("Promotions require a reference root, output, positive finite threshold and limit 1–11")
    print(json.dumps([asdict(run) for run in runs], indent=2), flush=True)
    if not args.execute:
        return
    if (not args.output or not args.gpu or any(not gpu or "," in gpu for gpu in args.gpu)
            or len(set(args.gpu)) != len(args.gpu) or args.batch_size < 1):
        parser.error("--execute requires an output, distinct GPUs and a positive batch size")
    if any(path is None or not path.is_file()
           for path in (args.manifest, args.clip_weights, args.dino_weights, *args.external_manifest)):
        parser.error("--execute requires existing data manifests and encoder weights")
    try:
        args.gpu = resolve_gpus(args.gpu)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = audit_isaid(args.manifest, args.output / "isaid_data_audit.json")
        results = run_queue([run for run in runs if run in FIXED], args)
        if args.promote_min_delta is not None:
            selected, decisions = promotions(args.output, args.reference_root,
                                             args.promote_min_delta, args.promotion_limit, args.ids, audit)
            write_json(args.output / "promotions.json", decisions)
            results.update(run_queue(selected, args))
        selected, blocked = extra_jobs(runs, args, audit)
        results.update(blocked)
        results.update(run_queue(selected, args))
        results.update(run_queue([run for run in runs if run in RESERVE], args))
        all_results = {path.parent.name: read_json(path)
                       for path in sorted(args.output.glob("*/queue_status.json"))}
        all_results.update(results)
        write_json(args.output / "queue_summary.json", all_results)
    raise SystemExit(any(status["state"] != "complete" for status in results.values()))


if __name__ == "__main__":
    main()
