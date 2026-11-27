import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from prepare_isaid import CLASSES
from run_v4_queue import (EXTRA, FIXED, GIB, RESERVE, Run, dev_result, extra_jobs, launch_phase,
                          optional_external_manifest_paths, promotions, resolve_gpus, resolve_promotions, run_job, run_queue)


def report(path, score, sample_hash="samples", epoch=15, *, legacy=False, batch=24, dino="pinned-dino", lr=.001):
    variant = path.parents[1].name
    data_hash = None if legacy else "audit"
    steps = ((18076 + batch - 1) // batch) * epoch
    training = {
        "config": {"variant": variant}, "manifest": "manifest", "data_sha256": data_hash,
        "size": 384, "seed": 42, "batch_size": batch, "lr": lr, "weight_decay": .01, "limit": None,
        "training_policy": {"epochs": epoch, "planned_steps": steps, "schedule": "cosine",
                            "warmup_steps": min(300, steps // 10), "min_lr_ratio": .1, "grad_clip": 1.},
        "weights": {"clip": "pinned-clip", "dinov3": dino}, "alignment_weight": .2,
    }
    if variant == "C2":
        training["config"].update(fusion="context", region_mode="component")
    if variant in ("F01", "F02") and not legacy:
        training["fusion_protocol"] = "unit_branch_cosine_v1"
    if variant == "F09" and not legacy:
        training["patch_to_text_loss_fraction"] = .5
    full = path.parent.name == "eval"
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    matrix[0][0], matrix[0][2] = score, 100 - score
    matrix[1][1], matrix[1][2] = score, 100 - score
    data = {**training,
            "checkpoint_steps": steps, "checkpoint_epoch": epoch, "checkpoint_kind": "evaluation_only",
            "checkpoint_train_manifest": "manifest", "sample_ids_sha256": sample_hash,
            "test_classes": CLASSES, "train_classes": CLASSES, "batch_size": 1,
            "prompt": "a photo of a {}.", "dtype": "float32", "tf32": False, "ignore_id": 255,
            "split": "val", "force_keep_all": False, "images": 6363 if full else 512,
            "eval_samples": None if full else 512, "eval_sample_seed": 42,
            "confusion_matrix": matrix, "class_iou": {CLASSES[0]: -1}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    if legacy:
        train_report = path.parents[1] / "train" / "report.json"
        train_report.parent.mkdir(parents=True, exist_ok=True)
        train_report.write_text(json.dumps({**training, "completed_epochs": epoch, "steps": steps}), encoding="utf-8")
    else:
        data["checkpoint_training"] = training
        data["checkpoint_sha256"] = "2" * 64
        path.write_text(json.dumps(data), encoding="utf-8")


class QueueTest(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual((len(FIXED), len(RESERVE)), (33, 12))
        self.assertEqual(sum(job.epochs == 40 for job in FIXED), 9)
        all_jobs = (*FIXED, *RESERVE, *[job for pair in EXTRA.values() for job in pair])
        self.assertEqual(len({job.id for job in all_jobs}), len(all_jobs))
        self.assertTrue({f"F{i:02d}" for i in range(1, 10)} <= {job.id for job in FIXED})

    def test_matched_promotion_uses_reports_and_keeps_pair(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs = root / "reference"
            report(refs / "B0/dev_epoch_015/report.json", 10, legacy=True)
            report(root / "F01/main/F01/dev_epoch_015/report.json", 12)
            second = root / "F02/main/F02/dev_epoch_015/report.json"
            report(second, 9, "wrong")
            report(root / "T01/main/B0/eval/report.json", 10, epoch=40)
            selected, decisions = promotions(root, refs, 1, 2)
            self.assertFalse(selected)
            self.assertTrue(any(d["state"] == "protocol_mismatch" for d in decisions))
            report(second, 9)
            selected, decisions = promotions(root, refs, 1, 1)
            self.assertFalse(selected)
            self.assertTrue(any(d["state"] == "budget" for d in decisions))
            selected, decisions = promotions(root, refs, 1, 2)
            self.assertEqual([r.id for r in selected], ["F01-e40", "F02-e40"])
            self.assertEqual(promotions(root, refs, 1, 2, ["F01"]), ([], []))
            self.assertEqual(promotions(root, refs, 1, 2, ["F01", "F02"])[0], selected)

    def test_promotion_requires_matched_training_and_full_anchor(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs = root / "reference"
            report(refs / "C2/dev_epoch_015/report.json", 10, legacy=True)
            candidate = root / "F06/main/C2/dev_epoch_015/report.json"
            long_anchor = root / "T04/main/C2/eval/report.json"

            def candidate_report(**kwargs):
                report(candidate, 12, **kwargs)
                data = json.loads(candidate.read_text())
                data["alignment_weight"] = 0.
                data["checkpoint_training"]["alignment_weight"] = 0.
                candidate.write_text(json.dumps(data))

            def decision():
                selected, decisions = promotions(root, refs, 1, 2, ["F06"])
                return selected, decisions[0]["state"]

            report(long_anchor, 10, epoch=40)
            for mismatch in ({"batch": 12}, {"dino": "different"}):
                candidate_report(**mismatch)
                self.assertEqual(decision(), ([], "protocol_mismatch"))
            candidate_report(lr=.1)
            self.assertEqual(decision(), ([], "invalid_report"))
            candidate_report()
            for mismatch in ({"batch": 12}, {"dino": "different"}, {"lr": .1}):
                report(long_anchor, 10, epoch=40, **mismatch)
                self.assertEqual(decision(), ([], "invalid_long_anchor"))
            report(long_anchor, 10, epoch=40)
            altered = json.loads(long_anchor.read_text())
            altered["images"] = 512
            long_anchor.write_text(json.dumps(altered))
            self.assertEqual(decision(), ([], "invalid_long_anchor"))
            report(long_anchor, 10, epoch=40)
            self.assertEqual(decision()[1], "selected")

    def test_diagnostic_inference_reports_are_not_promotion_evidence(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "B0/dev_epoch_015/report.json"
            report(path, 10)
            original = json.loads(path.read_text())
            for field, value in (("inference_prompts", ["a satellite view of {}"]),
                                 ("native_clip_weight", .25), ("inference_rotations", [0, 90]),
                                 ("inference_sizes", [512])):
                path.write_text(json.dumps({**original, field: value}))
                with self.assertRaisesRegex(ValueError, "Unexpected training or evaluation protocol"):
                    dev_result(path, Run("B0", "B0"))

    def test_factorial_promotes_both_missing_cells(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs = root / "reference"
            report(refs / "B0/dev_epoch_015/report.json", 10, legacy=True)
            report(root / "T01/main/B0/eval/report.json", 11, epoch=40)
            report(root / "T03/main/C1/eval/report.json", 12, epoch=40)
            report(root / "E01/main/E01/dev_epoch_015/report.json", 14)
            ids = ("E01", "E02")
            self.assertEqual(promotions(root, refs, 1, 2, ids)[1][0]["state"], "missing_reports")
            report(root / "E02/main/E02/dev_epoch_015/report.json", 9)
            selected, _ = promotions(root, refs, 1, 2, ids)
            self.assertEqual([run.id for run in selected], ["E01-e40", "E02-e40"])

    def test_extra_waits_for_matching_anchor_reports(self):
        with TemporaryDirectory() as folder:
            cases = EXTRA["resolution"]
            args = SimpleNamespace(output=Path(folder), batch_size=24,
                                   clip_weights="pinned-clip", dino_weights="pinned-dino")
            audit = {"source_manifest_sha256": "manifest", "data_sha256": "audit"}
            with patch("run_v4_queue.file_sha256", side_effect=lambda x: x):
                picked, blocked = extra_jobs(cases, args, audit)
                self.assertEqual(picked, [])
                self.assertEqual({anchor["anchor"] for anchor in blocked[cases[0].id]["anchors"]}, {"T01", "T03"})
                report(args.output / "T01/main/B0/eval/report.json", 15, epoch=40)
                c1 = args.output / "T03/main/C1/eval/report.json"
                for mismatch in ({"batch": 12}, {"dino": "wrong"}):
                    report(c1, 16, epoch=40, **mismatch)
                    self.assertEqual(extra_jobs(cases, args, audit)[0], [])
                report(c1, 16, epoch=40)
                self.assertEqual(extra_jobs(cases, args, audit), (list(cases), {}))

    def test_gpu_index_alias_is_not_a_second_device(self):
        inventory = SimpleNamespace(stdout="0, GPU-abc\n1, GPU-def\n")
        with patch("run_v4_queue.subprocess.run", return_value=inventory):
            self.assertEqual(resolve_gpus(["1", "GPU-abc"]), ["GPU-def", "GPU-abc"])
            with self.assertRaisesRegex(ValueError, "distinct physical GPUs"):
                resolve_gpus(["0", "GPU-abc"])

    def test_failed_job_does_not_drop_other_jobs(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), gpu=["a", "b"])
            runs = FIXED[:4]

            def execute(run, gpu, options):
                self.assertIn(gpu, options.gpu)
                (options.output / run.id).mkdir()
                if run == runs[0]:
                    raise ValueError("fixture error")
                return {"state": "complete"}

            with patch("run_v4_queue.run_job", side_effect=execute):
                results = run_queue(runs, args)
            self.assertEqual(results[runs[0].id]["state"], "failed")
            self.assertTrue(all(results[run.id]["state"] == "complete" for run in runs[1:]))
            self.assertEqual(len(list(args.output.glob("*/queue_status.json"))), len(runs))

    def test_reserve_fills_free_lane_while_fixed_job_still_runs(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), gpu=["a", "b"])
            long, short, reserve = FIXED[0], FIXED[1], RESERVE[0]
            long_started, reserve_started = Event(), Event()

            def execute(run, gpu, options):
                if run == long:
                    long_started.set()
                    if not reserve_started.wait(4):
                        raise RuntimeError("reserve was blocked by the fixed-stage barrier")
                if run == reserve:
                    if not long_started.wait(4):
                        raise RuntimeError("did not dispatch the fixed job first")
                    reserve_started.set()
                return {"state": "complete"}

            with patch("run_v4_queue.run_job", side_effect=execute):
                results = run_queue((long, short, reserve), args)
            self.assertTrue(all(status["state"] == "complete" for status in results.values()))
            state = json.loads((args.output / "queue_live.json").read_text())
            self.assertEqual(state["active_by_gpu"], {})
            self.assertEqual(state["terminal"], {run.id: "complete" for run in (long, short, reserve)})
            summary = json.loads((args.output / "queue_summary.json").read_text())
            self.assertEqual(summary[reserve.id]["category"], "reserve")

    def test_preflight_uses_both_devices_and_never_starts_main(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), gpu=["a", "b"], manifest="manifest",
                                   clip_weights="clip", dino_weights="dino", batch_size=24, retries=0,
                                   preflight_only=True, min_free_gib=20.)
            first, second = Event(), Event()

            def launch(command, env, log, stage_path, stage, retries):
                self.assertEqual((command[2], command[-1], stage["phase"]), ("preflight", "24", "preflight"))
                self.assertEqual(env["CSERIES_MIN_FREE_BYTES"], str(20 * GIB))
                if env["CUDA_VISIBLE_DEVICES"] == "a":
                    first.set()
                    self.assertTrue(second.wait(4))
                    return False, [{"retryable": False}]
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "b")
                second.set()
                self.assertTrue(first.wait(4))
                return True, []

            runs = (EXTRA["resolution"][0], FIXED[0])
            with patch("run_v4_queue.disk_usage", return_value=SimpleNamespace(free=32 * GIB)):
                with patch("run_v4_queue.launch_phase", side_effect=launch) as invoke:
                    results = run_queue(runs, args)
            self.assertEqual(invoke.call_count, 2)
            self.assertEqual(results[runs[0].id]["state"], "failed")
            self.assertEqual(results[runs[1].id]["state"], "complete")
            self.assertTrue((args.output / "queue_preflight_summary.json").exists())
            self.assertFalse((args.output / "queue_summary.json").exists())

    def test_disk_floor_records_blocked_admission_then_allows_resume(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), gpu=["a", "b"], min_free_gib=20.)
            runs = FIXED[:2]
            with patch("run_v4_queue.run_task", return_value={"state": "complete"}) as run:
                with patch("run_v4_queue.disk_usage", return_value=SimpleNamespace(free=19 * GIB)):
                    results = run_queue(runs, args)
                self.assertFalse(run.called)
                for item in results.values():
                    self.assertEqual(item["reason"], "disk_low_watermark")
                live = json.loads((args.output / "queue_live.json").read_text())
                self.assertTrue(live["admission_paused"])
                self.assertEqual(live["disk"]["min_free_bytes"], 20 * GIB)
                self.assertEqual(live["ready"], [run.id for run in runs])
                with patch("run_v4_queue.disk_usage", return_value=SimpleNamespace(free=21 * GIB)):
                    results = run_queue(runs, args)
                self.assertEqual(run.call_count, 2)
                self.assertTrue(all(item["state"] == "complete" for item in results.values()))
                live = json.loads((args.output / "queue_live.json").read_text())
                self.assertFalse(live["admission_paused"])

    def test_history_interleaves_with_training_on_two_devices(self):
        with TemporaryDirectory() as folder:
            histories = [SimpleNamespace(id=f"HIST-{variant}") for variant in ("C0", "B0")]
            args = SimpleNamespace(output=Path(folder), gpu=["a", "b"], history_runs=histories)
            starting = Event()
            training = FIXED[0]

            def execute(task, gpu, options):
                if task.category == "fixed":
                    self.assertEqual(gpu, "b")
                    starting.set()
                if task.id == histories[0].id:
                    self.assertEqual(gpu, "a")
                    if not starting.wait(4):
                        raise RuntimeError("History tasks displaced both training lanes")
                return {"state": "complete"}

            with patch("run_v4_queue.run_task", side_effect=execute):
                results = run_queue((training,), args)
            self.assertEqual(len(results), 3)
            self.assertTrue(all(item["state"] == "complete" for item in results.values()))

    def test_optional_diagnostic_depends_only_on_its_anchor(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), gpu=["a", "b"], diagnostics=True)
            anchor = next(run for run in FIXED if run.id == "T01")
            other = FIXED[0]
            diagnostic_started = Event()

            def execute(task, gpu, options):
                if task.id == other.id and not diagnostic_started.wait(4):
                    raise RuntimeError("diagnostic waited on unrelated job")
                if task.id == "I-T01":
                    diagnostic_started.set()
                return {"state": "complete"}

            with patch("run_v4_queue.run_task", side_effect=execute):
                results = run_queue((anchor, other), args)
            self.assertEqual(set(results), {"T01", other.id, "I-T01"})
            self.assertTrue(all(status["state"] == "complete" for status in results.values()))

    def test_transient_attempt_retries_but_deterministic_failure_does_not(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for reason, expected in (("Stale file handle", 2), ("CUDA out of memory", 1)):
                with self.subTest(reason=reason):
                    marker = root / reason
                    log = root / "log.txt"
                    program = (
                        "import sys; from pathlib import Path; marker = Path(sys.argv[1]); "
                        "known = marker.exists(); marker.touch(); print(sys.argv[2]); "
                        "raise SystemExit(0 if known else 1)"
                    )
                    with patch("run_v4_queue.sleep") as delay:
                        done, failures = launch_phase(
                            [sys.executable, "-c", program, str(marker), reason], os.environ,
                            log, root / "stage.json", {"phase": "main"}, retries=1,
                        )
                    self.assertEqual(done, expected == 2)
                    self.assertEqual(delay.call_count, expected - 1)
                    self.assertEqual(failures[0]["retryable"], expected == 2)
                    self.assertEqual(json.loads((root / "stage.json").read_text())["attempt"], expected)

    def test_retry_does_not_relaunch_below_disk_floor(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            env = {**os.environ, "CSERIES_MIN_FREE_BYTES": str(20 * GIB)}
            with patch("run_v4_queue.disk_usage", side_effect=[
                SimpleNamespace(free=21 * GIB), SimpleNamespace(free=19 * GIB),
            ]):
                with patch("run_v4_queue.sleep"):
                    done, failures = launch_phase(
                        [sys.executable, "-c", "raise SystemExit(75)"], env,
                        root / "log.txt", root / "stage.json", {"phase": "main"}, retries=1,
                    )
            self.assertFalse(done)
            self.assertEqual((failures[0]["returncode"], failures[-1]["reason"]), (75, "disk_low_watermark"))
            self.assertEqual((root / "log.txt").read_text().count("[queue]"), 1)
            self.assertEqual(json.loads((root / "stage.json").read_text())["phase"], "paused_disk")

    def test_external_stage_failure_does_not_fail_source_training(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output=root, manifest="manifest", clip_weights="clip", dino_weights="dino",
                                   batch_size=24, external_manifest=["bad-external", "good-external"], retries=1)
            with patch("run_v4_queue.launch_phase", side_effect=[
                (True, []), (True, []), (False, [{"manifest": "bad-external"}]), (True, [])
            ]) as launch:
                result = run_job(next(run for run in FIXED if run.id == "T01"), "gpu", args)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["external_failures"], [{"manifest": "bad-external"}])
            self.assertEqual(launch.call_count, 4)
            self.assertEqual(launch.call_args_list[2].args[1]["KEEP_FINAL_STATE"], "1")
            self.assertEqual(launch.call_args_list[2].args[1]["CSERIES_TRACK_DEVELOPMENT"], "1")

    def test_missing_optional_manifest_is_separate_from_valid_inputs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, valid, missing = (root / name for name in ("source.json", "valid.json", "missing.json"))
            source.write_text('{"train": [], "val": []}')
            valid.write_text('{"val": [{"sample_id": "test", "file_name": "im.png", "sem_seg_file_name": "mask.png"}]}')
            accepted, skipped = optional_external_manifest_paths(source, [valid, missing, valid])
            self.assertEqual(accepted, [valid])
            self.assertEqual([entry["path"] for entry in skipped], [str(missing)])

    def test_resolved_promotion_plan_is_replayed(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), reference_root=Path(folder) / "reference",
                                   promote_min_delta=1., promotion_limit=2, ids=["F01", "F02"],
                                   promotion_external_manifest=[], promotion_class_groups=None)
            audit = {"source_manifest_sha256": "manifest", "data_sha256": "audit"}
            selected = [Run("F01-e40", "F01", 40), Run("F02-e40", "F02", 40)]
            with patch("run_v4_queue.promotions", return_value=(selected, [{"state": "selected"}])) as choose:
                self.assertEqual(resolve_promotions(args, audit), selected)
                self.assertEqual(resolve_promotions(args, audit), selected)
            self.assertEqual(choose.call_count, 1)
            args.promote_min_delta = 2.
            with self.assertRaisesRegex(ValueError, "Promotion protocol changed"):
                resolve_promotions(args, audit)

    def test_promotion_rejects_mislabeled_c2_control(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs = root / "reference"
            baseline = refs / "C2/dev_epoch_015/report.json"
            candidate = root / "F03/main/C2/dev_epoch_015/report.json"
            for path, score in ((baseline, 10), (candidate, 12)):
                report(path, score)
                data = json.loads(path.read_text())
                data["config"].update(fusion="context", region_mode="component")
                data["alignment_weight"] = .2
                path.write_text(json.dumps(data))
            for variant, fusion in (("F04", "learned"), ("F05", "separate")):
                path = root / variant / "main/C2/dev_epoch_015/report.json"
                report(path, 13)
                data = json.loads(path.read_text())
                data["config"]["fusion"] = data["checkpoint_training"]["config"]["fusion"] = fusion
                path.write_text(json.dumps(data))
            selected, decisions = promotions(root, refs, 1, 2, ["F03", "F04", "F05"])
            self.assertFalse(selected)
            self.assertEqual(decisions[0]["cells"][0]["state"], "invalid_report")

    def test_f08_promotion_requires_correct_forward_weight(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "F08/dev_epoch_015/report.json"
            report(path, 12)
            data = json.loads(path.read_text())
            data["alignment_weight"] = .2
            for fraction in (None, 1., .5):
                data["region_to_text_loss_fraction"] = fraction
                data["checkpoint_training"]["region_to_text_loss_fraction"] = fraction
                path.write_text(json.dumps(data))
                if fraction == .5:
                    self.assertEqual(dev_result(path, Run("F08", "F08"))[3], 12)
                else:
                    with self.assertRaisesRegex(ValueError, "matched F08"):
                        dev_result(path, Run("F08", "F08"))


if __name__ == "__main__":
    unittest.main()
