from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

from isaid_protocol import file_sha256
from v4_diagnostics import CASES, run_diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_reuses_only_matching_checkpoint_and_diagnostic_options(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            root = output / "T03"
            checkpoint = root / "main/C1/train/final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture")
            (output / "isaid_data_audit.json").write_text(json.dumps({
                "source_manifest_sha256": "source", "data_sha256": "content",
            }))
            args = SimpleNamespace(output=output, manifest=output / "manifest.json",
                                   clip_weights=output / "clip.pt", dino_weights=output / "dino.pt")
            run = SimpleNamespace(id="T03", variant="C1", epochs=40, seed=42)
            cases = CASES[:2]

            def invoke(command, **_options):
                dest = command[command.index("--output") + 1]
                _, _, sizes, angles, prompts, weight = next(case for case in cases if case[0] == dest.name)
                (dest / "report.json").write_text(json.dumps({
                    "checkpoint_sha256": file_sha256(checkpoint), "manifest": "source",
                    "data_sha256": "content", "checkpoint_epoch": 40, "images": 512,
                    "eval_samples": 512, "eval_sample_seed": 42, "sample_ids_sha256": "same-subset",
                    "inference_sizes": sizes, "inference_rotations": angles, "inference_prompts": prompts,
                    "native_clip_weight": weight, "miou_percent": 40.0, "ms_per_image": 1.0,
                }))

            with patch("v4_diagnostics.CASES", cases), patch("v4_diagnostics.subprocess.run", side_effect=invoke) as launch:
                self.assertEqual(run_diagnostics(run, "gpu", args)["state"], "complete")
                self.assertEqual(run_diagnostics(run, "gpu", args)["state"], "complete")
                self.assertEqual(launch.call_count, 2)
                checkpoint.write_bytes(b"changed checkpoint")
                changed = run_diagnostics(run, "gpu", args)
                self.assertEqual(changed["state"], "failed")
                self.assertEqual(launch.call_count, 2)
                self.assertIn("checkpoint or protocol", changed["cases"]["baseline"]["reason"])
