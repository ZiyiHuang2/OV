from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from prepare_isaid import CLASSES
from v4_history import audit_split, class_errors, history_runs, run_history


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class HistoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = self.root = Path(temporary.name)
        source = root / "source.json"
        external = root / "external.json"
        write_json(root / "source_classes.json", CLASSES)
        write_json(root / "external_classes.json", ["buildings", "cars", "ship"])
        write_json(source, {"test_classes": "source_classes.json", "val": [
            {"sample_id": "sample-2"}, {"sample_id": "sample-1"}], "ignore_label": 255})
        write_json(external, {"test_classes": "external_classes.json", "val": [
            {"sample_id": "x17"}, {"sample_id": "x12"}], "ignore_label": 255})
        self.source_hash = sha256(source.read_bytes()).hexdigest()
        self.external_hash = sha256(external.read_bytes()).hexdigest()
        self.data_hash = "d" * 64
        self.args = SimpleNamespace(manifest=source, external_manifest=[external],
                                    output=root / "output", reference_root=root / "historical_main",
                                    clip_weights=root / "clip.pt", dino_weights=root / "dino.pt")
        write_json(self.args.output / "isaid_data_audit.json",
                   {"source_manifest_sha256": self.source_hash, "data_sha256": self.data_hash})

    def archive(self, variant):
        weights = {"clip": "CLIP"}
        if variant != "C0":
            weights["dinov3"] = "DINOv3"
        training = {
            "schema": 2, "config": {"variant": variant}, "weights": weights, "size": 384,
            "prompt": "a photo of a {}.", "manifest": self.source_hash, "train_classes": CLASSES,
            "ignore_id": 255, "seed": 42, "batch_size": 24, "lr": .001, "weight_decay": .01,
            "split": "train", "limit": None, "dtype": "float32", "tf32": False,
            "steps": 11310, "completed_epochs": 15, "alignment_weight": .2,
            "training_policy": {"epochs": 15, "planned_steps": 11310, "schedule": "cosine",
                                "warmup_steps": 300, "min_lr_ratio": .1, "grad_clip": 1.},
        }
        dest = self.args.reference_root / variant / "train"
        write_json(dest / "report.json", training)
        (dest / "final.pt").write_bytes(b"immutable test checkpoint")
        return training

    def report(self, run, training):
        classes = CLASSES if run.source else ["buildings", "cars", "ship"]
        samples = "sample-2\nsample-1\n" if run.source else "x17\nx12\n"
        report = {
            **training, "manifest": run.manifest_sha256, "checkpoint_train_manifest": self.source_hash,
            "checkpoint_sha256": sha256(b"immutable test checkpoint").hexdigest(),
            "checkpoint_kind": "evaluation_only", "checkpoint_epoch": 15, "checkpoint_steps": 11310,
            "checkpoint_training": training, "test_classes": classes, "images": 2, "split": "val",
            "sample_ids_sha256": sha256(samples.encode()).hexdigest(), "batch_size": 1,
            "eval_samples": None, "force_keep_all": False, "inference_sizes": [384],
            "inference_rotations": [0], "inference_prompts": ["a photo of a {}."],
            "native_clip_weight": float(run.native_clip), "miou_percent": 45.,
            "confusion_matrix": [[int(i == j) for j in range(len(classes))] for i in range(len(classes))],
        }
        if run.source:
            report["data_sha256"] = self.data_hash
        return report

    def test_plan_uses_only_real_and_distinct_evaluation_entries(self):
        duplicate = self.root / "external-copy.json"
        duplicate.write_bytes(self.args.external_manifest[0].read_bytes())
        self.args.external_manifest.extend([duplicate, self.root / "missing.json"])
        runs = history_runs(self.args)
        self.assertEqual([r.id for r in runs[:2]], ["HIST-iSAID-C0", "HIST-iSAID-native-CLIP"])
        self.assertEqual([r.variant for r in runs[2:]], ["C0", "B0", "C1", "D0", "C2"])
        self.assertTrue(all(r.manifest_sha256 == self.external_hash for r in runs[2:]))
        self.assertFalse((self.args.output / "history").exists())

    def test_native_mode_and_external_anchor_store_and_reuse_report(self):
        runs = history_runs(self.args)
        native, external_b0 = runs[1], runs[3]
        for run in (native, external_b0):
            with self.subTest(job=run.id):
                training = self.archive(run.variant)
                evaluation = self.report(run, training)
                relative = (Path("history/isaid_val/native-CLIP/report.json") if run.source else
                            Path("history/external") / self.external_hash[:16] / "B0/eval/report.json")
                out = self.args.output / relative

                def invoke(command, **kwargs):
                    self.assertIn("--eval-loss", command)
                    self.assertEqual("--native-clip-weight" in command, run.native_clip)
                    self.assertEqual("--dino-weights" in command, run.variant != "C0")
                    self.assertEqual("--data-sha256" in command, run.source)
                    self.assertEqual(command[command.index("--output") + 1], out.parent)
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "gpu2")
                    write_json(out, evaluation)

                with patch("v4_history.subprocess.run", side_effect=invoke) as launch:
                    first = run_history(run, "gpu2", self.args)
                    self.assertEqual(first["state"], "complete", first)
                    self.assertEqual(first, run_history(run, "gpu2", self.args))
                    self.assertEqual(launch.call_count, 1)
                    self.assertEqual(first["report"], str(relative))
                    self.assertFalse((self.args.reference_root / "external").exists())
                    self.assertTrue(out.with_name("class_errors.json").is_file())
                    write_json(out, {**evaluation, "sample_ids_sha256": "wrong samples"})
                    mismatch = run_history(run, "gpu2", self.args)
                    self.assertEqual(mismatch["state"], "failed")
                    self.assertIn("checkpoint or protocol", mismatch["reason"])
                    self.assertEqual(launch.call_count, 1)

    def test_missing_checkpoint_and_mutated_source_are_isolated(self):
        run = history_runs(self.args)[0]
        result = run_history(run, "gpu2", self.args)
        self.assertEqual(result["state"], "blocked")
        self.archive("C0")
        with run.manifest.open("a") as file:
            file.write("\n")
        result = run_history(run, "gpu2", self.args)
        self.assertEqual(result["state"], "failed")
        self.assertIn("manifest changed", result["reason"])

    def test_published_split_requires_audited_manifest_identity(self):
        source, _, external = history_runs(self.args)[:3]
        self.assertEqual(audit_split(source), "iSAID_split/val")
        self.assertIsNone(audit_split(external))
        audit = external.manifest.with_name("data_audit.json")
        write_json(audit, {"manifest_sha256": "different", "published_split": "DLRSD_split/val"})
        self.assertIsNone(audit_split(external))
        write_json(audit, {"manifest_sha256": self.external_hash, "published_split": "DLRSD_split/val"})
        self.assertEqual(audit_split(external), "DLRSD_split/val")

    def test_class_errors_preserve_vocabulary_and_ground_truth_orientation(self):
        summary = class_errors({"test_classes": ["Building", "cars", "ship"],
                                "confusion_matrix": [[5, 2, 3], [1, 6, 0], [3, 1, 4]]})
        self.assertEqual(summary["focal_classes"], {"buildings": ["Building"], "vehicles": ["cars"]})
        self.assertEqual(summary["class_errors"]["Building"]["false_negative_pixels"], 5)
        self.assertEqual(summary["class_errors"]["Building"]["missed_as"], {"cars": 2, "ship": 3})
        self.assertEqual(summary["class_errors"]["cars"]["false_positive_pixels"], 3)
        self.assertEqual(summary["class_errors"]["cars"]["false_positive_from"], {"Building": 2, "ship": 1})
        self.assertEqual(summary["class_errors"]["cars"]["precision"], 6 / 9)
        with self.assertRaisesRegex(ValueError, "Confusion matrix"):
            class_errors({"test_classes": ["Building", "cars"], "confusion_matrix": [[2, -1], [0, 1]]})
