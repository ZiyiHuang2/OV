"""Exercise the real Bash launcher in a temporary repository with stub training jobs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


MOCK = """
import argparse, json, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('mode')
p.add_argument('--output', type=Path)
p.add_argument('--manifest', type=Path)
p.add_argument('--variant')
p.add_argument('--epochs', type=int)
p.add_argument('--batch-size', type=int)
p.add_argument('--checkpoint', type=Path)
a, _ = p.parse_known_args()
a.output.mkdir(parents=True, exist_ok=True)
manifest = json.loads(a.manifest.read_text())
if a.mode == 'train':
    for name in ['state', 'final'] + (['epoch_001', 'epoch_003'] if a.epochs == 5 else []):
        (a.output / (name + '.pt')).write_text(name)
    report = {'config': {'variant': a.variant}, 'completed_epochs': a.epochs,
              'steps': a.epochs * 3, 'training_policy': {'epochs': a.epochs}}
else:
    assert a.checkpoint.is_file(), a.checkpoint
    variant = a.checkpoint.parent.parent.name
    epoch = int(a.checkpoint.stem[-3:]) if a.checkpoint.stem.startswith('epoch') else (1 if 'preflight' in a.checkpoint.parts else 5)
    bad = os.environ.get('BAD_STAGE') == a.output.name
    report = {'config': {'variant': variant}, 'checkpoint_epoch': epoch - bad,
              'checkpoint_kind': 'evaluation_only', 'checkpoint_steps': epoch * 3,
              'images': len(manifest['val']), 'miou_percent': 1}
(a.output / 'report.json').write_text(json.dumps(report))
"""


def verify(launcher):
    with tempfile.TemporaryDirectory(prefix="cseries-shell-test-") as tmp:
        root = Path(tmp)
        shutil.copyfile(launcher, root / "run_isaid_suite.sh")
        (root / "prepare_isaid.py").write_text("CLASSES = ['mock']\nSPLIT_SIZES = {'train': 2, 'val': 1}\n")
        (root / "run_combinations.py").write_text(MOCK)
        (root / "classes.json").write_text('["mock"]')
        manifest = {"ignore_label": 255, "train_classes": "classes.json", "test_classes": "classes.json"}
        for split, samples in (("train", ("tr0", "tr1")), ("val", ("va0",))):
            manifest[split] = []
            for sample in samples:
                row = {"sample_id": sample}
                for sub, key in (("image", "file_name"), ("mask", "sem_seg_file_name")):
                    path = root / sub / f"{sample}.png"
                    path.parent.mkdir(exist_ok=True)
                    path.touch()
                    row[key] = str(path.relative_to(root))
                manifest[split].append(row)
        for name in ("clip.pt", "dino.pt"):
            (root / name).touch()
        good = root / "manifest.json"
        good.write_text(json.dumps(manifest))
        partial = root / "partial.json"
        partial.write_text(json.dumps({**manifest, "train": manifest["train"][:1]}))
        for args in (("init", "-q"), ("add", "."), ("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")):
            subprocess.run(["git", *args], cwd=root, check=True)

        def run(phase, manifest=good, dest="run", **env):
            result = subprocess.run(["bash", "run_isaid_suite.sh", phase, str(manifest), "clip.pt", "dino.pt", dest, "8"],
                                    cwd=root, env={**os.environ, "PYTHON": sys.executable, **env},
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return result.returncode, result.stdout

        failures = []
        if not run("preflight", partial, "partial_out", PYTHONOPTIMIZE="1")[0] or (root / "partial_out/preflight.ok").exists():
            failures.append("partial manifest passed the formal gate under PYTHONOPTIMIZE=1")
        wrong_names = root / "wrong_names.json"
        bad_rows = [{**row, "sem_seg_file_name": "mask/tr1.png"} for row in manifest["train"]]
        wrong_names.write_text(json.dumps({**manifest, "train": bad_rows}))
        if not run("preflight", wrong_names, "wrong_names_out")[0]:
            failures.append("a mask with the wrong sample name passed the formal gate")
        code, log = run("preflight")
        assert not code, log
        assert not list((root / "run/preflight").glob("**/*.pt"))
        train = root / "run/main/C0/train"
        train.mkdir(parents=True)
        state = train / "state.pt"
        state.touch()
        report = train / "report.json"
        report.write_text(json.dumps({"config": {"variant": "C0"}, "completed_epochs": 1, "steps": 4}))
        code, log = run("main")
        if not code or "resume explicitly" not in log:
            failures.append("interrupted training report was accepted by run_job:\n" + log)
        assert state.exists() and not (train / "final.pt").exists()
        for name in ("epoch_001", "epoch_003", "final"):
            (train / f"{name}.pt").write_text(name)
        report.write_text(json.dumps({"config": {"variant": "C0"}, "completed_epochs": 5, "steps": 15}))
        code, log = run("main", BAD_STAGE="dev_epoch_003")
        assert code, log
        assert not (train / "epoch_001.pt").exists()
        assert all((train / f"{name}.pt").exists() for name in ("epoch_003", "final", "state"))
        code, log = run("main")
        assert not code, "\n".join(failures + ["invalid eval report prevented retry:", log])
        assert sorted(p.name for p in (root / "run/main").glob("**/*.pt")) == ["final.pt"] * 9
        assert (root / "run/main/C6/keep_all/report.json").is_file()
        assert (root / "run/main/summary.csv").is_file()
        code, log = run("preflight", dest="keep")
        assert not code, log
        code, log = run("main", dest="keep", KEEP_FINAL_STATE="1")
        assert not code, log
        assert sorted(p.name for p in (root / "keep/main").glob("**/*.pt")) == ["final.pt"] * 9 + ["state.pt"] * 9
    assert not failures, failures
    print("PASS: full-data gate under -O, preflight cleanup, incomplete-train resume gate, failed-eval retention, retry and KEEP_FINAL_STATE")


if __name__ == "__main__":
    launcher = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("run_isaid_suite.sh")
    verify(launcher.resolve())
