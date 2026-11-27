import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from isaid_protocol import HELD_CLASSES
from prepare_isaid import CLASSES
from run_v4_queue import FIXED
from v4_decisions import promotion_candidates, promotion_dependencies, report_path, select_promotions


def save(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row), encoding="utf-8")
    return path


def report(variant, scores, epochs=15):
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    for i, score in enumerate(scores):
        matrix[i][i] = score
        matrix[i][len(scores)] = 100 - score
    train = {"manifest": "manifest", "data_sha256": "audit", "config": {"variant": variant},
             "weights": {"clip": "pinned-clip", "dinov3": "pinned-dino"}, "seed": 42,
             "size": 384, "batch_size": 24, "lr": .001, "weight_decay": .01, "limit": None,
             "training_policy": {"epochs": epochs}, "alignment_weight": .2}
    return {**train, "checkpoint_training": train, "checkpoint_epoch": epochs,
            "checkpoint_steps": epochs * 754, "checkpoint_kind": "evaluation_only",
            "checkpoint_sha256": sha256(variant.encode()).hexdigest(), "checkpoint_train_manifest": "manifest",
            "split": "val", "batch_size": 1, "dtype": "float32", "tf32": False, "force_keep_all": False,
            "prompt": "a photo of a {}.", "sample_ids_sha256": "sample-order",
            "images": 512, "eval_samples": 512, "eval_sample_seed": 42, "ignore_id": 255,
            "test_classes": CLASSES, "train_classes": CLASSES, "confusion_matrix": matrix,
            "score": sum(scores) / len(scores)}


def dev_result(path, run, development=True):
    row = json.loads(path.read_text())
    if row["checkpoint_training"]["config"]["variant"] != run.variant or row["checkpoint_epoch"] != run.epochs:
        raise ValueError("Fixture doesn't match run")
    key = {name: row[name] for name in ("manifest", "weights")}
    subset = row["sample_ids_sha256"], [sum(line) for line in row["confusion_matrix"]]
    return key, subset, row["data_sha256"], row["score"]


class PromotionTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "archive" / "main"

    def rows(self, ids, anchor, short_scores, long_id, anchor_scores=(70,)):
        cells = {}
        for name, scores in zip(ids, short_scores, strict=True):
            if scores is not None:
                cells[name] = report(name, scores)
                save(report_path(self.root, name, name), cells[name])
        save(self.reference / anchor / "dev_epoch_015" / "report.json", report(anchor, anchor_scores))
        save(report_path(self.root, long_id, anchor, "eval"), report(anchor, anchor_scores, 40))
        return cells

    def decide(self, ids, limit=11, **kwargs):
        ranked, decisions = promotion_candidates(self.root, self.reference, 1., FIXED, dev_result, ids, **kwargs)
        picked, decisions = select_promotions(ranked, decisions, FIXED, limit)
        return [run.id for run in picked], decisions[0]

    def test_transfer_evidence_has_priority_when_budget_is_limited(self):
        signals = [
            (18., ("D01",), {"trigger": "source_dev_miou", "transfer_priority": False}),
            (3., ("D02",), {"trigger": "held_dev_hiou", "transfer_priority": True}),
        ]
        chosen, decisions = select_promotions(signals, [], FIXED, limit=1)
        self.assertEqual([run.id for run in chosen], ["D02-e40"])
        self.assertEqual([row["state"] for row in decisions], ["selected", "budget"])

    def test_shared_gain_cannot_justify_an_early_late_comparison(self):
        pair = ("F01", "F02")
        self.rows(pair, "B0", ((75,), (75,)), "T01")
        picked, decision = self.decide(pair)
        self.assertFalse(picked)
        self.assertEqual(decision["state"], "below_threshold")
        self.assertEqual(decision["delta_pp"], 0)
        self.assertEqual(decision["evidence"][0]["baseline_deltas_pp"], {"F01": 5, "F02": 5})

    def test_large_negative_performance_can_leave_a_meaningful_contrast(self):
        pair = ("F01", "F02")
        self.rows(pair, "B0", ((50,), (55,)), "T01")
        self.assertEqual(self.decide(pair, 1)[1]["state"], "budget")
        picked, decision = self.decide(pair)
        self.assertEqual(picked, ["F01-e40", "F02-e40"])
        self.assertEqual(decision["delta_pp"], 5)
        self.assertEqual(decision["priority_pp"], 5)
        self.assertEqual(decision["evidence"][0]["baseline_deltas_pp"], {"F01": -20, "F02": -15})
        self.rows(pair, "B0", ((55,), (50,)), "T01")
        self.assertEqual(self.decide(pair)[1]["delta_pp"], -5)

    def test_patch_and_region_controls_cannot_be_split(self):
        pair = ("F08", "F09")
        self.rows(pair, "C2", (None, (75,)), "T04")
        picked, decision = self.decide(pair)
        self.assertFalse(picked)
        self.assertEqual(decision["state"], "missing_reports")
        self.assertEqual(decision["ids"], pair)
        self.assertEqual(promotion_dependencies(["F09"]), set())
        self.assertEqual(promotion_dependencies(pair), {"F08", "F09", "T04"})

        self.rows(pair, "C2", ((60,), (75,)), "T04")
        self.assertEqual(self.decide(pair, 1)[1]["state"], "budget")
        self.assertEqual(self.decide(pair)[0], ["F08-e40", "F09-e40"])

    def test_reverse_alignment_contrast_keeps_patch_pair_coherent(self):
        pair = ("F08", "F09")
        self.rows(pair, "C2", ((60,), (60,)), "T04")
        picked, decision = self.decide(pair)
        self.assertEqual(picked, ["F08-e40", "F09-e40"])
        self.assertEqual(decision["delta_pp"], 0)
        self.assertEqual(decision["trigger"], "source_dev_reverse_alignment")
        self.assertEqual(decision["priority_pp"], 10)

    def test_short_factorial_interaction_can_justify_the_missing_cells(self):
        pair = ("E01", "E02")
        self.rows(pair, "B0", ((75,), (75,)), "T01")
        save(report_path(self.root, "T03", "C1", "eval"), report("C1", (80,), 40))
        short_c1 = self.reference / "C1" / "dev_epoch_015" / "report.json"
        row = report("C1", (90,))
        self.assertFalse(self.decide(pair)[0])
        save(short_c1, {**row, "sample_ids_sha256": "not-matched"})
        self.assertFalse(self.decide(pair)[0])
        save(short_c1, row)
        picked, decision = self.decide(pair)
        self.assertEqual(picked, ["E01-e40", "E02-e40"])
        self.assertEqual(decision["trigger"], "source_dev_factorial_interaction")
        self.assertEqual(decision["priority_pp"], 10)

    def test_registered_class_group_can_justify_a_tied_pair(self):
        pair = ("F01", "F02")
        self.rows(pair, "B0", ((80, 60), (60, 80)), "T01", (70, 70))
        self.assertEqual(self.decide(pair)[1]["state"], "below_threshold")
        picked, decision = self.decide(pair, class_groups={"from-plan": CLASSES[:1]})
        self.assertEqual(picked, ["F01-e40", "F02-e40"])
        self.assertEqual(decision["trigger"], "source_class_miou")
        self.assertEqual(decision["priority_pp"], 20)
        self.assertEqual(self.decide(pair, class_groups={"unsupported": [CLASSES[2]]})[1]["evidence"][1]["state"],
                         "missing_class_support")

    def test_external_development_requires_bound_checkpoints_and_matched_samples(self):
        pair = ("F01", "F02")
        cells = self.rows(pair, "B0", ((75,), (75,)), "T01")
        manifest = save(self.root / "external_dev" / "manifest.json",
                        {"test_classes": "classes.json", "ignore_label": 255,
                         "val": [{"sample_id": "v01"}]})
        save(manifest.parent / "classes.json", CLASSES)
        digest = sha256(manifest.read_bytes()).hexdigest()
        for name, score in zip(pair, (35, 55), strict=True):
            external = {**cells[name],
                        "manifest": digest, "sample_ids_sha256": sha256(b"v01\n").hexdigest(),
                        "images": 1, "eval_samples": None, "eval_sample_seed": None,
                        "confusion_matrix": report(name, (score,))["confusion_matrix"]}
            path = self.root / name / "external" / digest[:16] / name / "eval" / "report.json"
            save(path, external)
        self.assertEqual(self.decide(pair)[1]["state"], "below_threshold")
        picked, decision = self.decide(pair, external_manifests=[manifest])
        self.assertEqual(picked, ["F01-e40", "F02-e40"])
        self.assertEqual(decision["trigger"], "external_dev_miou")
        self.assertEqual(decision["priority_pp"], 20)
        for key, value in (("checkpoint_sha256", "wrong"), ("sample_ids_sha256", "other")):
            save(path, {**external, key: value})
            picked, decision = self.decide(pair, external_manifests=[manifest])
            self.assertFalse(picked)
            self.assertEqual(decision["evidence"][-1]["state"], "invalid_report")

    def test_held_development_uses_seen_and_held_on_same_split(self):
        pair = ("F01", "F02")
        size = len(CLASSES) - 1
        scores = (75,) * size
        cells = self.rows(pair, "B0", (scores, scores), "T01", scores)
        held_indices = [CLASSES.index(category) for category in HELD_CLASSES["H1"]]
        seen = [category for category in CLASSES if category not in HELD_CLASSES["H1"]]
        for name, hit_rate in zip(pair, (25, 75), strict=True):
            job = name + "-H1"
            rates = tuple(hit_rate if i in held_indices else 75 for i in range(size))
            derived = save(self.root / job / "held_data" / "manifest.json",
                           {"source_manifest_sha256": "manifest", "source_data_sha256": "audit"})
            digest = sha256(derived.read_bytes()).hexdigest()
            path = report_path(self.root, job, name)
            final = path.parents[1] / "train" / "final.pt"
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_bytes(name.encode())
            held = {**cells[name], "manifest": digest, "checkpoint_train_manifest": digest, "train_classes": seen,
                    "checkpoint_training": {**cells[name]["checkpoint_training"], "manifest": digest},
                    "confusion_matrix": report(name, rates)["confusion_matrix"]}
            save(path, held)

        picked, decision = self.decide(pair)
        self.assertEqual(picked, ["F01-e40", "F02-e40"])
        self.assertEqual(decision["trigger"], "held_dev_hiou")
        self.assertIn("held_miou", decision["evidence"][1]["breakdown"]["F02"])
        save(path, {**held, "sample_ids_sha256": "nonmatched"})
        self.assertFalse(self.decide(pair)[0])
        self.assertEqual(promotion_dependencies(pair), {"F01", "F02", "T01", "F01-H1", "F02-H1"})


if __name__ == "__main__":
    unittest.main()
