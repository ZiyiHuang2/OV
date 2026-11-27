import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from prepare_isaid import CLASSES
from run_v4_queue import EXTRA, FIXED, RESERVE, Run, dev_result, extra_jobs, promotions, resolve_gpus, run_queue


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
    full = path.parent.name == "eval"
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    matrix[0][:2] = [score, 100 - score]
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
            report(refs / "B0/dev_epoch_015/report.json", 10, legacy=True)
            candidate = root / "D01/main/D01/dev_epoch_015/report.json"
            long_anchor = root / "T01/main/B0/eval/report.json"

            def decision():
                selected, decisions = promotions(root, refs, 1, 2, ["D01"])
                return selected, decisions[0]["state"]

            report(long_anchor, 10, epoch=40)
            for mismatch in ({"batch": 12}, {"dino": "different"}):
                report(candidate, 12, **mismatch)
                self.assertEqual(decision(), ([], "protocol_mismatch"))
            report(candidate, 12, lr=.1)
            self.assertEqual(decision(), ([], "invalid_report"))
            report(candidate, 12)
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
            selected, decisions = promotions(root, refs, 1, 2)
            self.assertFalse(selected)
            self.assertEqual(next(d for d in decisions if d["ids"] == ("F03",))["state"], "invalid_report")

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
