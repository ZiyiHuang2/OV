"""Resolve and run the matched iSAID experiment queue through the formal launcher."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import errno
import json
import math
import os
from pathlib import Path
import re
from shutil import disk_usage
from statistics import median
import subprocess
import sys
from time import monotonic, sleep, time

from isaid_protocol import audit_isaid, check_external_source, file_sha256
from prepare_isaid import CLASSES, SPLIT_SIZES


HERE = Path(__file__).resolve().parent
FORMAL_PROMPT = "a photo of a {}."
GIB = 1 << 30
DEFAULT_MIN_FREE_GIB = 20.


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
DIAGNOSTIC_ANCHORS = ("T01", "T03")
PRIORITY = {"history": 0, "preflight": 0, "fixed": 1, "promotion": 2, "extra": 3, "reserve": 4, "diagnostic": 5}
NON_TRAINING = frozenset(("history", "diagnostic", "preflight"))
TRANSIENT_ERRNOS = {errno.EAGAIN, errno.EBUSY, errno.EIO, errno.ESTALE, errno.ETIMEDOUT,
                    errno.ECONNRESET, errno.ENETDOWN}
TRANSIENT_LOG = re.compile(
    r"stale file handle|resource temporarily unavailable|input/output error|transport endpoint is not connected"
    r"|connection (?:reset by peer|timed out)|temporarily unavailable",
    re.IGNORECASE,
)
TERMINAL_LOG = re.compile(
    r"out of memory|protocol changed|resume mismatch|unexpected .*sha-256|nonfinite (?:loss|evaluation|gradient)",
    re.IGNORECASE,
)
LOG_WINDOW = 32 << 10



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
    if run.variant in ("F01", "F02") and (data.get("fusion_protocol"), training.get("fusion_protocol")) != (
            "unit_branch_cosine_v1", "unit_branch_cosine_v1"):
        raise ValueError(f"Expected matched unit-branch fusion protocol: {path}")
    if run.variant == "F09" and (
            data.get("patch_to_text_loss_fraction"), data["alignment_weight"],
            training.get("patch_to_text_loss_fraction")) != (.5, .2, .5):
        raise ValueError(f"Expected matched F09 patch-to-text weight: {path}")
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


def promotions(root, reference_root, minimum_delta, limit, requested_ids=None, audit=None, *,
               external_manifests=(), class_groups=None, historical_evidence_root=None):
    from v4_decisions import promotion_candidates, select_promotions

    ranked, decisions = promotion_candidates(
        root, reference_root, minimum_delta, FIXED, dev_result, requested_ids, audit,
        external_manifests=external_manifests, class_groups=class_groups,
        historical_evidence_root=historical_evidence_root,
    )
    return select_promotions(ranked, decisions, FIXED, limit)


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


def read_tail(path, start=0):
    with path.open("rb") as stream:
        end = stream.seek(0, os.SEEK_END)
        stream.seek(max(end - LOG_WINDOW, start, 0))
        return stream.read().decode("utf-8", errors="replace")


def launch_phase(command, env, log_path, stage_path, stage, retries):
    failures = []
    floor = int(env.get("CSERIES_MIN_FREE_BYTES", 0))
    for attempt in range(1, retries + 2):
        current = {**stage, "attempt": attempt, "started_at": time(), "pid": None}
        disk = disk_state(stage_path.parent, floor)
        if disk["blocked"]:
            failure = {**stage, "attempt": attempt, "reason": "disk_low_watermark", "disk": disk}
            write_json(stage_path, {**current, "phase": "paused_disk", "last_failure": failure})
            return False, [*failures, failure]
        write_json(stage_path, current)
        try:
            with log_path.open("ab") as log:
                log.write((f"\n[queue] {stage['phase']} attempt {attempt}\n").encode())
                log.flush()
                start = log.tell()
                with subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
                    write_json(stage_path, {**current, "pid": process.pid})
                    returncode = process.wait()
            if returncode == 0:
                return True, failures
            tail = read_tail(log_path, start)
            retryable = returncode == 75 or bool(TRANSIENT_LOG.search(tail) and not TERMINAL_LOG.search(tail))
            failure = {"returncode": returncode, "retryable": retryable}
        except OSError as exc:
            retryable = exc.errno in TRANSIENT_ERRNOS
            failure = {"reason": str(exc), "retryable": retryable}
        failures.append({**stage, "attempt": attempt, **failure})
        if not retryable or attempt > retries:
            return False, failures
        write_json(stage_path, {**current, "phase": "retry_wait", "pid": None, "last_failure": failures[-1]})
        sleep(min(5 * attempt, 15))


def disk_state(path, floor):
    free = disk_usage(path).free
    return {"free_bytes": free, "min_free_bytes": floor, "blocked": free < floor}


def disk_floor(args):
    return int(getattr(args, "min_free_gib", 0) * GIB)


def queue_artifact(args, name):
    mode = "queue_preflight" if getattr(args, "preflight_only", False) else "queue"
    return f"{mode}_{name}.json"


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
           "CSERIES_SIZE": str(run.size), "CSERIES_UNATTENDED": "1", "CSERIES_TRACK_DEVELOPMENT": "1",
           "KEEP_FINAL_STATE": "0" if getattr(args, "discard_final_state", False) else "1",
           "CSERIES_DATA_AUDIT": str(args.output / "isaid_data_audit.json"),
           "CSERIES_MIN_FREE_BYTES": str(disk_floor(args))}
    launcher = ["bash", str(HERE / "run_isaid_suite.sh")]
    operands = [str(p) for p in (args.manifest, args.clip_weights, args.dino_weights, out)] + [str(args.batch_size)]
    preflight_only = getattr(args, "preflight_only", False)
    stages = [("preflight", None)]
    if not preflight_only:
        stages.append(("main", None))
        if not run.split:
            stages += [("external", path) for path in args.external_manifest]
    attempts, external_failures = [], []
    for phase, external in stages:
        disk = disk_state(args.output, disk_floor(args))
        if disk["blocked"]:
            return {"state": "blocked", "reason": "disk_low_watermark", "next_phase": phase,
                    "disk": disk, "attempt_failures": attempts, "external_failures": external_failures}
        phase_env = {**env, "EXTERNAL_MANIFEST": str(external or "")}
        stage = {"phase": phase, "manifest": str(external) if external else None}
        success, failures = launch_phase(
            [*launcher, phase, *operands], phase_env, out / "queue.log", out / "queue_stage.json",
            stage, args.retries,
        )
        attempts.extend(failures)
        if not success:
            disk = disk_state(args.output, disk_floor(args))
            if disk["blocked"] or failures[-1].get("reason") == "disk_low_watermark":
                return {"state": "blocked", "reason": "disk_low_watermark", "next_phase": phase,
                        "disk": failures[-1].get("disk", disk), "attempt_failures": attempts,
                        "external_failures": external_failures}
            if phase != "external":
                return {"state": "failed", "failed_phase": phase, "attempt_failures": attempts}
            external_failures.append(failures[-1])
    return {"state": "complete", "attempt_failures": attempts, "external_failures": external_failures}


@dataclass(frozen=True)
class Task:
    run: Run
    category: str

    @property
    def id(self):
        return f"I-{self.run.id}" if self.category == "diagnostic" else self.run.id


def run_task(task, gpu, args):
    if task.category == "history":
        from v4_history import run_history

        return run_history(task.run, gpu, args)
    if task.category == "diagnostic":
        from v4_diagnostics import run_diagnostics

        return run_diagnostics(task.run, gpu, args)
    return run_job(task.run, gpu, args)


def optional_external_manifest_paths(source, paths):
    accepted, skipped = [], []
    for path in dict.fromkeys(paths):
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing manifest: {path}")
            check_external_source(source, path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
        else:
            accepted.append(path)
    return accepted, skipped


def task_progress(task, args):
    if task.category in ("history", "diagnostic"):
        return {}
    out = args.output / task.run.id
    try:
        stage = read_json(out / "queue_stage.json")
    except FileNotFoundError:
        return {}
    if stage["phase"] == "main":
        train_dir = out / "main" / task.run.variant / "train"
        planned = math.ceil(SPLIT_SIZES["train"] / args.batch_size) * task.run.epochs
        try:
            report = read_json(train_dir / "report.json")
            steps = report["steps"]
        except FileNotFoundError:
            try:
                tail = read_tail(train_dir / "console.log")
                observed = re.findall(r'(?:^step=|"steps":\s*)(\d+)', tail, re.MULTILINE)
                steps = max(map(int, observed), default=0)
            except FileNotFoundError:
                steps = 0
        stage.update(steps=steps, planned_steps=planned)
    return stage


def epoch_work(task):
    return task.run.epochs * (task.run.size / 384) ** 2


def live_state(args, ready, active, waiting, results, timings):
    now = monotonic()
    lanes, estimates = {}, []
    rate = median(timings) if timings else None
    for _, (task, gpu, started) in active.items():
        elapsed = now - started
        progress = task_progress(task, args)
        lanes[gpu] = {"id": task.id, "category": task.category, "elapsed_seconds": round(elapsed, 1), **progress}
        if rate is not None and task.category not in NON_TRAINING:
            estimates.append(max(rate * epoch_work(task) - elapsed, 0.))
    eta = None
    if rate is not None:
        queued = sum(epoch_work(task) for task in ready if task.category not in NON_TRAINING) * rate
        eta = round(max(max(estimates, default=0.), (queued + sum(estimates)) / len(args.gpu)))
    return {"updated_at_unix": time(), "active_by_gpu": lanes, "ready": [task.id for task in ready],
            "waiting_on_evidence": waiting,
            "terminal": {name: status["state"] for name, status in results.items()},
            "eta_seconds_for_ready_training": eta, "eta_observed_jobs": len(timings),
            "skipped_external_manifests": getattr(args, "skipped_external_manifests", [])}


def resolve_promotions(args, audit):
    # Freeze the science decision before admitting work, so a restart cannot
    # choose different groups or spend the budget twice.
    path = args.output / "promotion_plan.json"
    protocol = {"source_manifest_sha256": audit["source_manifest_sha256"],
                "data_sha256": audit["data_sha256"], "reference_root": str(args.reference_root),
                "threshold_pp": args.promote_min_delta, "limit": args.promotion_limit,
                "requested_ids": sorted(args.ids or []),
                "external_manifests": [str(path) for path in args.promotion_external_manifest],
                "historical_evidence_root": str(args.output / "history" / "external"),
                "class_groups": args.promotion_class_groups}
    if path.is_file():
        plan = read_json(path)
        if plan["protocol"] != protocol:
            raise ValueError("Promotion protocol changed; use a new output directory")
        selected, decisions = [Run(**values) for values in plan["selected"]], plan["decisions"]
    else:
        selected, decisions = promotions(
            args.output, args.reference_root, args.promote_min_delta, args.promotion_limit, args.ids, audit,
            external_manifests=args.promotion_external_manifest, class_groups=args.promotion_class_groups,
            historical_evidence_root=args.output / "history" / "external",
        )
        write_json(path, {"protocol": protocol, "selected": [asdict(run) for run in selected],
                          "decisions": decisions})
    write_json(args.output / "promotions.json", decisions)
    return selected


def run_queue(runs, args, *, audit=None):
    from v4_decisions import promotion_dependencies

    preflight_only = getattr(args, "preflight_only", False)
    ready = [Task(run, "preflight" if preflight_only else "reserve" if run in RESERVE else "fixed")
             for run in runs if preflight_only or not any(run in pair for pair in EXTRA.values())]
    extra = {group: [run for run in pair if run in runs]
             for group, pair in EXTRA.items() if not preflight_only and any(run in runs for run in pair)}
    diagnostics = {run.id: run for run in runs if not preflight_only and getattr(args, "diagnostics", False)
                   and run.id in DIAGNOSTIC_ANCHORS}
    enabled = {task.id for task in ready}
    pending_promotion = not preflight_only and getattr(args, "promote_min_delta", None) is not None
    promotion_needs = promotion_dependencies(getattr(args, "ids", None)) & enabled if pending_promotion else set()
    histories = getattr(args, "history_runs", ()) if not preflight_only else ()
    if pending_promotion:
        promotion_needs.update(
            run.id for run in histories if not run.source and run.variant in ("C0", "B0", "C2")
            and run.manifest in args.promotion_external_manifest
        )
    ready.extend(Task(run, "history") for run in histories)
    results, active, timings = {}, {}, []
    all_results = {}
    status_name = queue_artifact(args, "status")
    summary_path = args.output / queue_artifact(args, "summary")
    live_path = args.output / queue_artifact(args, "live")
    for path in sorted(args.output.glob(f"*/{status_name}")):
        try:
            all_results[path.parent.name] = read_json(path)
        except (OSError, ValueError) as exc:
            all_results[path.parent.name] = {"state": "unreadable_status", "reason": str(exc)}

    def finish(task_id, status):
        try:
            directory = args.output / task_id
            directory.mkdir(exist_ok=True)
            write_json(directory / status_name, status)
        except OSError as exc:
            status = {"state": "failed", "reason": f"Cannot write queue status: {exc}", "job_status": status}
        results[task_id] = all_results[task_id] = status
        write_json(summary_path, all_results)

    with ThreadPoolExecutor(max_workers=len(args.gpu)) as pool:
        while True:
            if pending_promotion and promotion_needs <= results.keys():
                ready.extend(Task(run, "promotion") for run in resolve_promotions(args, audit))
                pending_promotion = False
            for group, group_runs in list(extra.items()):
                if (set(EXTRA_ANCHORS[group]) & enabled) <= results.keys():
                    selected, blocked = extra_jobs(group_runs, args, audit)
                    ready.extend(Task(run, "extra") for run in selected)
                    for task_id, status in blocked.items():
                        finish(task_id, status)
                    del extra[group]
            for anchor, run in list(diagnostics.items()):
                if anchor in results:
                    task = Task(run, "diagnostic")
                    if results[anchor]["state"] == "complete":
                        ready.append(task)
                    else:
                        finish(task.id, {"state": "blocked", "anchor": anchor})
                    del diagnostics[anchor]

            ready.sort(key=lambda task: PRIORITY[task.category])
            occupied = {gpu for _, gpu, _ in active.values()}
            disk = disk_state(args.output, disk_floor(args))
            for gpu in (gpu for gpu in args.gpu if gpu not in occupied):
                if not ready or disk["blocked"]:
                    break
                # Reserve one lane for new training while history runs in parallel.
                history_active = any(task.category == "history" for task, _, _ in active.values())
                index = next((i for i, task in enumerate(ready) if task.category != "history"), 0) if history_active else 0
                task = ready.pop(index)
                started = monotonic()
                active[pool.submit(run_task, task, gpu, args)] = task, gpu, started
                disk = disk_state(args.output, disk_floor(args))
            waiting = {run.id: sorted(set(EXTRA_ANCHORS[group]) - results.keys())
                       for group, group_runs in extra.items() for run in group_runs}
            waiting.update({f"I-{anchor}": [anchor] for anchor in diagnostics})
            if pending_promotion:
                waiting["promotions"] = sorted(promotion_needs - results.keys())
            if ready and disk["blocked"] and not active:
                for task in ready:
                    finish(task.id, {"state": "blocked", "category": task.category,
                                     "reason": "disk_low_watermark", "disk": disk})
            write_json(live_path, {**live_state(args, ready, active, waiting, results, timings),
                                   "disk": disk, "admission_paused": bool(ready and disk["blocked"])})
            if not active:
                break

            done, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                task, gpu, started = active.pop(future)
                elapsed = monotonic() - started
                try:
                    status = future.result()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    status = {"state": "failed", "reason": str(exc)}
                status.update(category=task.category, gpu=gpu, wall_seconds=round(elapsed, 2))
                if status["state"] == "complete" and task.category not in NON_TRAINING and elapsed >= 60:
                    timings.append(elapsed / epoch_work(task))
                finish(task.id, status)
                print(f"{task.id} {status['state']} on {gpu}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "clip-weights", "dino-weights", "output", "reference-root"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve())
    parser.add_argument("--external-manifest", action="append", type=lambda p: Path(p).resolve(), default=[])
    parser.add_argument("--gpu", action="append", help="Repeat for each GPU UUID or index")
    parser.add_argument("--batch-size", type=int, default=24)
    reserve = parser.add_mutually_exclusive_group()
    reserve.add_argument("--include-reserve", action="store_true", help="Include reserves in a dry run")
    reserve.add_argument("--no-reserve", action="store_true", help="Exclude reserves from execution")
    parser.add_argument("--diagnostics", action="store_true", help="Run inference diagnostics after T01/T03")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--history", action="store_true", help="Evaluate historical checkpoints alongside new training")
    modes.add_argument("--history-only", action="store_true", help="Evaluate historical checkpoints without training")
    modes.add_argument("--preflight-only", action="store_true", help="Run the full-data CUDA preflight only")
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB,
                        help="Pause new work below this free disk space on the output filesystem (default: 20 GiB)")
    parser.add_argument("--discard-final-state", action="store_true", help="Remove the completed optimizer state")
    parser.add_argument("--retries", type=int, default=1, help="Extra attempts for transient stage failures, 0–3")
    parser.add_argument("--extra", choices=EXTRA, action="append", default=[])
    parser.add_argument("--ids", nargs="+", help="Restrict to these exact job IDs")
    parser.add_argument("--promote-min-delta", type=float, help="Enable matched 40-epoch follow-ups at this threshold in pp")
    parser.add_argument("--promotion-limit", type=int, default=11)
    parser.add_argument("--promotion-external-manifest", action="append", type=lambda p: Path(p).resolve(),
                        default=[], help="A preregistered external development manifest also listed for evaluation")
    parser.add_argument("--promotion-class-groups", type=Path,
                        help="JSON object mapping preregistered group names to iSAID class lists")
    parser.add_argument("--execute", action="store_true", help="Start jobs, otherwise print the resolved plan")
    args = parser.parse_args()
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error("--min-free-gib must be finite and non-negative")
    if (args.history or args.history_only) and args.reference_root is None:
        parser.error("--history and --history-only require --reference-root")
    if (args.history_only or args.preflight_only) and (args.promote_min_delta is not None or args.diagnostics):
        parser.error("Promotion and diagnostics require the training queue")
    use_reserves = args.include_reserve or (args.execute and not args.no_reserve)
    runs = [*FIXED, *(RESERVE if use_reserves else ())]
    runs.extend(run for name in dict.fromkeys(args.extra) for run in EXTRA[name])
    if args.history_only:
        runs = []
    if args.ids:
        if len(set(args.ids)) != len(args.ids) or set(args.ids) - {r.id for r in runs}:
            parser.error("--ids must list distinct IDs from the enabled queue")
        runs = [run for run in runs if run.id in args.ids]
    if args.promote_min_delta is not None:
        if (args.reference_root is None or args.output is None or not math.isfinite(args.promote_min_delta)
                or args.promote_min_delta <= 0 or not 1 <= args.promotion_limit <= 11):
            parser.error("Promotions require a reference root, output, positive finite threshold and limit 1–11")
    if set(args.promotion_external_manifest) - set(args.external_manifest):
        parser.error("Promotion external manifests must also be listed for evaluation")
    if (args.promotion_external_manifest or args.promotion_class_groups) and args.promote_min_delta is None:
        parser.error("Promotion evidence requires --promote-min-delta")
    if args.promotion_class_groups:
        try:
            args.promotion_class_groups = read_json(args.promotion_class_groups)
            if (not isinstance(args.promotion_class_groups, dict)
                    or any(not name or not isinstance(classes, list) or not classes
                           or any(not isinstance(c, str) or c not in CLASSES for c in classes)
                           or len(set(classes)) != len(classes)
                           for name, classes in args.promotion_class_groups.items())):
                raise ValueError("Expected unique iSAID class names in nonempty JSON groups")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    def print_plan(history=()):
        if history:
            plan = {"training": [asdict(run) for run in runs], "history": [asdict(run) for run in history]}
        else:
            plan = [asdict(run) for run in runs]
        print(json.dumps(plan, indent=2, default=str), flush=True)

    if not args.execute:
        if args.history or args.history_only:
            from v4_history import history_runs

            print_plan(history_runs(args))
        else:
            print_plan()
        return
    if (not args.output or not args.gpu or any(not gpu or "," in gpu for gpu in args.gpu)
            or len(set(args.gpu)) != len(args.gpu) or args.batch_size < 1):
        parser.error("--execute requires an output, distinct GPUs and a positive batch size")
    if not 0 <= args.retries <= 3:
        parser.error("--retries must range from 0 to 3")
    if any(path is None or not path.is_file() for path in (args.manifest, args.clip_weights, args.dino_weights)):
        parser.error("--execute requires an existing source data manifest and encoder weights")
    try:
        args.gpu = resolve_gpus(args.gpu)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        disk = disk_state(args.output, disk_floor(args))
        if disk["blocked"]:
            write_json(args.output / queue_artifact(args, "live"),
                       {"updated_at_unix": time(), "admission_paused": True, "disk": disk,
                        "ready": [run.id for run in runs], "active_by_gpu": {}})
            raise SystemExit(f"Output filesystem below the free-space floor: {disk['free_bytes'] / GIB:.1f} GiB free")
        audit = audit_isaid(args.manifest, args.output / "isaid_data_audit.json")
        args.external_manifest, args.skipped_external_manifests = optional_external_manifest_paths(
            args.manifest, args.external_manifest)
        for skipped in args.skipped_external_manifests:
            print(f"Skipped optional external manifest {skipped['path']}: {skipped['reason']}", flush=True)
        write_json(args.output / "external_manifest_status.json",
                   {"accepted": [str(path) for path in args.external_manifest], "skipped": args.skipped_external_manifests})
        if args.history or args.history_only:
            from v4_history import history_runs

            args.history_runs = history_runs(args)
        print_plan(getattr(args, "history_runs", ()))
        results = run_queue(runs, args, audit=audit)
    raise SystemExit(any(status.get("reason") == "disk_low_watermark"
                         or (status["state"] == "failed" and status.get("category") not in ("diagnostic", "history"))
                         or (args.history_only and status["state"] != "complete")
                         for status in results.values()))



if __name__ == "__main__":
    main()
