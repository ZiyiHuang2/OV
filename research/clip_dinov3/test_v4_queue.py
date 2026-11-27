import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_v4_queue import EXTRA, FIXED, RESERVE, Run, dev_result, promotions, run_queue


def report(path, score, sample_hash="samples", epoch=15):
    data = {
        "checkpoint_epoch": epoch, "checkpoint_kind": "evaluation_only",
        "manifest": "manifest", "sample_ids_sha256": sample_hash,
        "test_classes": ["a", "b"], "train_classes": ["a", "b"],
        "size": 384, "seed": 42, "ignore_id": 255, "training_policy": {"epochs": epoch},
        "config": {"variant": path.parents[1].name},
        "weights": {"clip": "pinned"}, "confusion_matrix": [[10, 0], [0, 0]],
        "class_iou": {"a": score, "b": 0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
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
            report(refs / "B0/dev_epoch_015/report.json", 10)
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
                path.write_text(json.dumps(data))
                if fraction == .5:
                    self.assertEqual(dev_result(path, Run("F08", "F08"))[1], 12)
                else:
                    with self.assertRaisesRegex(ValueError, "matched F08"):
                        dev_result(path, Run("F08", "F08"))


if __name__ == "__main__":
    unittest.main()
