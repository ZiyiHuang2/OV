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


HERE = Path(__file__).resolve().parent


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
    *held_pair("H1"), *held_pair("H2"),
    *held_pair("H1", seed=43), *held_pair("H2", seed=43),
    *(Run(v, v) for v in ("E01", "E02", "D01", "D02")),
    *(Run(f"T0{i}", v, 40) for i, v in ((1, "B0"), (3, "C1"))),
    *(Run(v, v) for v in ("F01", "F02")),
    *(Run(f"T0{i}", v, 40) for i, v in ((0, "C0"), (2, "D0"), (4, "C2"))),
    *(Run(f"F{i:02d}", "C2", control=c) for i, c in enumerate(
        ("mean", "learned", "separate", "noalign", "classpool"), 3)),
    Run("F08", "F08"), Run("F09", "F09"),
    *(Run(f"{v}-H1", v, split="H1") for v in ("D0", "C1", "C2")),
    *held_pair("H1", 40), *held_pair("H2", 40),
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


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def dev_result(path, run):
    data = read_json(path)
    config = data["config"]
    if (config["variant"] != run.variant or data["checkpoint_epoch"] != 15
            or data["checkpoint_kind"] != "evaluation_only"):
        raise ValueError(f"Unexpected variant or checkpoint: {path}")
    if run.variant == "C2":
        fusion = run.control if run.control in ("mean", "learned", "separate") else "context"
        region = "class" if run.control == "classpool" else "component"
        alignment = 0 if run.control == "noalign" else .2
        if (config["fusion"], config["region_mode"], data["alignment_weight"]) != (fusion, region, alignment):
            raise ValueError(f"Unexpected C2 control: {path}")
    populations = [sum(row) for row in data["confusion_matrix"]]
    scores = [data["class_iou"][name] for name, support in
              zip(data["test_classes"], populations, strict=True) if support]
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError(f"No valid supported-class score: {path}")
    identity = tuple(data[key] for key in ("manifest", "sample_ids_sha256", "test_classes",
                                          "train_classes", "size", "seed", "ignore_id",
                                          "training_policy")) + (populations, data["weights"]["clip"])
    return identity, sum(scores) / len(scores)


def promotions(root, reference_root, minimum_delta, limit):
    fixed = {run.id: run for run in FIXED}
    long_anchor = {"C0": "T00", "B0": "T01", "C2": "T04"}
    groups = [(("D01",), "B0"), (("D02",), "C0"), (("E01",), "B0"), (("E02",), "B0"),
              (("F01", "F02"), "B0")]
    groups += [((f"F{i:02d}",), "C2") for i in range(3, 10)]
    eligible, decisions = [], []
    for ids, anchor in groups:
        reference = reference_root / anchor / "dev_epoch_015" / "report.json"
        rows = [root / name / "main" / fixed[name].variant / "dev_epoch_015" / "report.json" for name in ids]
        decision = {"ids": ids, "anchor": anchor}
        if not reference.is_file() or not all(row.is_file() for row in rows):
            decisions.append({**decision, "state": "missing_reports"})
            continue
        try:
            anchor_key, anchor_score = dev_result(reference, Run(anchor, anchor))
            results = [dev_result(row, fixed[name]) for name, row in zip(ids, rows, strict=True)]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            decisions.append({**decision, "state": "invalid_report", "reason": str(exc)})
            continue
        if any(key != anchor_key for key, _ in results):
            decisions.append({**decision, "state": "protocol_mismatch"})
            continue
        required = [(long_anchor[anchor], anchor)]
        if ids[0] in ("E01", "E02"):
            required.append(("T03", "C1"))
        long_reports = [root / run_id / "main" / variant / "eval" / "report.json"
                        for run_id, variant in required]
        if not all(path.is_file() for path in long_reports):
            decisions.append({**decision, "state": "missing_long_anchor"})
            continue
        fields = ("manifest", "test_classes", "train_classes", "size", "seed", "ignore_id")
        try:
            source = read_json(reference)
            anchors = [read_json(path) for path in long_reports]
            valid = all(row["checkpoint_epoch"] == 40
                        and row["checkpoint_kind"] == "evaluation_only"
                        and row["training_policy"]["epochs"] == 40
                        and row["config"]["variant"] == variant
                        and all(row[key] == source[key] for key in fields)
                        for row, (_, variant) in zip(anchors, required, strict=True))
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            decisions.append({**decision, "state": "invalid_long_anchor"})
            continue
        delta = max(score for _, score in results) - anchor_score
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
           "CSERIES_SIZE": str(run.size), "CSERIES_UNATTENDED": "1"}
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
            results[run.id] = status
            print(f"{run.id} {status['state']} on {gpu}", flush=True)
            directory = args.output / run.id
            if directory.is_dir():
                write_json(directory / "queue_status.json", status)

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
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = run_queue(runs, args)
        if args.promote_min_delta is not None:
            selected, decisions = promotions(args.output, args.reference_root,
                                             args.promote_min_delta, args.promotion_limit)
            write_json(args.output / "promotions.json", decisions)
            results.update(run_queue(selected, args))
        all_results = {path.parent.name: read_json(path)
                       for path in sorted(args.output.glob("*/queue_status.json"))}
        write_json(args.output / "queue_summary.json", all_results)
    raise SystemExit(any(status["state"] != "complete" for status in results.values()))


if __name__ == "__main__":
    main()
