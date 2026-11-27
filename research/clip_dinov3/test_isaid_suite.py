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
from isaid_protocol import development_epochs

p = argparse.ArgumentParser()
p.add_argument('mode')
p.add_argument('--output', type=Path)
p.add_argument('--manifest', type=Path)
p.add_argument('--variant')
p.add_argument('--epochs', type=int)
p.add_argument('--batch-size', type=int)
p.add_argument('--checkpoint', type=Path)
p.add_argument('--seed', type=int)
p.add_argument('--eval-loss', action='store_true')
a, _ = p.parse_known_args()
a.output.mkdir(parents=True, exist_ok=True)
manifest = json.loads(a.manifest.read_text())
if a.mode == 'train':
    stages = [e for e in development_epochs(a.epochs) if e < a.epochs]
    if os.environ.get('FAIL_TRAIN_VARIANT') == a.variant:
        (a.output / 'state.pt').write_text('interrupted')
        raise SystemExit(9)
    if a.checkpoint is not None:
        assert a.checkpoint.read_text() == 'interrupted'
        print('fixture resumed')
    for name in ['state', 'final'] + [f'epoch_{e:03d}' for e in stages]:
        (a.output / (name + '.pt')).write_text(name)
    report = {'config': {'variant': a.variant}, 'completed_epochs': a.epochs,
              'steps': a.epochs * 3, 'training_policy': {'epochs': a.epochs},
              'batch_size': a.batch_size, 'seed': a.seed, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}
else:
    assert a.checkpoint.is_file(), a.checkpoint
    variant = a.checkpoint.parent.parent.name
    epoch = (int(a.checkpoint.stem[-3:]) if a.checkpoint.stem.startswith('epoch')
             else json.loads((a.checkpoint.parent / 'report.json').read_text())['completed_epochs'])
    bad = os.environ.get('BAD_STAGE') == a.output.name
    report = {'config': {'variant': variant}, 'checkpoint_epoch': epoch - bad,
              'checkpoint_kind': 'evaluation_only', 'checkpoint_steps': epoch * 3,
              'images': len(manifest['val']), 'miou_percent': 1,
              'test_classes': ['mock'], 'class_iou': {'mock': 2 if epoch == 18 else 1},
              'confusion_matrix': [[1]], 'sample_ids_sha256': 'fixture'}
    if a.eval_loss:
        report['mean_pixel_cross_entropy'] = 0.3
(a.output / 'report.json').write_text(json.dumps(report))
"""


def verify(launcher):
    with tempfile.TemporaryDirectory(prefix="cseries-shell-test-") as tmp:
        root = Path(tmp)
        shutil.copyfile(launcher, root / "run_isaid_suite.sh")
        shutil.copyfile(launcher.with_name("isaid_protocol.py"), root / "isaid_protocol.py")
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

        def run(phase, manifest=good, dest="run", batch=8, **env):
            result = subprocess.run(["bash", "run_isaid_suite.sh", phase, str(manifest), "clip.pt", "dino.pt", dest, str(batch)],
                                    cwd=root, env={**os.environ, "PYTHON": sys.executable, "CSERIES_EPOCHS": "5",
                                                   "CSERIES_GPU_IDS": "", "KEEP_FINAL_STATE": "0",
                                                   "CSERIES_VARIANTS": "", "CSERIES_SEED": "42",
                                                   "CSERIES_HELD_SPLIT": "", "CSERIES_UNATTENDED": "0", **env},
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
        long = {"CSERIES_EPOCHS": "15", "CSERIES_GPU_IDS": "gpu-a,gpu-b"}
        code, log = run("main", **long)
        assert code and "Protocol changed" in log, log
        code, log = run("preflight", dest="long", batch=24, **long)
        assert not code, log
        code, log = run("main", dest="long", batch=24, **long)
        assert not code, log
        run_dir = root / "long/main"
        assert sorted(p.name for p in run_dir.glob("**/*.pt")) == ["final.pt"] * 9
        for gpu, variants in (("gpu-a", ("C0", "D0", "C2", "C3", "C5")),
                              ("gpu-b", ("B0", "C1", "C4", "C6"))):
            for v in variants:
                model = run_dir / v
                report = json.loads((model / "train/report.json").read_text())
                assert (report["gpu"], report["batch_size"], report["completed_epochs"]) == (gpu, 24, 15)
                assert {d.name for d in model.glob("dev_epoch_*")} == {"dev_epoch_005", "dev_epoch_010", "dev_epoch_015"}
        assert "dev_epoch_010" in (run_dir / "summary.csv").read_text()
        assert (run_dir / "C6/keep_all/report.json").is_file()
        tracked = {"CSERIES_EPOCHS": "40", "CSERIES_VARIANTS": "C0,B0",
                   "CSERIES_SEED": "43", "CSERIES_UNATTENDED": "1"}
        code, log = run("preflight", dest="tracked", **tracked)
        assert not code, log
        code, log = run("main", dest="tracked", FAIL_TRAIN_VARIANT="C0", **tracked)
        assert code and "Failed: main/C0" in log, log
        tracked_dir = root / "tracked/main"
        assert (tracked_dir / "B0/eval/report.json").is_file()
        assert (tracked_dir / "C0/train/state.pt").read_text() == "interrupted"
        wrong_seed = {**tracked, "CSERIES_SEED": "44"}
        code, log = run("main", dest="tracked", **wrong_seed)
        assert code and "Protocol changed" in log, log
        code, log = run("main", dest="tracked", **tracked)
        assert not code and "fixture resumed" in log, log
        for variant in ("C0", "B0"):
            run_root = tracked_dir / variant
            assert sorted(p.name for p in run_root.glob("train/*.pt")) == [
                "best.pt", "epoch_015.pt", "epoch_030.pt", "final.pt"]
            assert (run_root / "train/best.pt").read_text() == "epoch_018"
            assert json.loads((run_root / "train/best.json").read_text())["epoch"] == 18
            assert json.loads((run_root / "train/report.json").read_text())["seed"] == 43
        code, log = run("main", dest="tracked", **tracked)
        assert not code, log
        assert "mean_pixel_cross_entropy" in (tracked_dir / "summary.csv").read_text()
        external = root / "external.json"
        external.write_text(json.dumps(manifest))
        code, log = run("external", dest="tracked", EXTERNAL_MANIFEST=str(external), **tracked)
        assert not code, log
        external_summary, = (root / "tracked/external").glob("*/summary.csv")
        assert "epoch_015" in external_summary.read_text() and "epoch_030" in external_summary.read_text()
        factorial = {"CSERIES_VARIANTS": "E01,E02", "CSERIES_GPU_IDS": "gpu-a,gpu-b"}
        code, log = run("preflight", dest="factorial", **factorial)
        assert not code, log
        code, log = run("main", dest="factorial", **factorial)
        assert not code, log
        factorial_dir = root / "factorial/main"
        for v, gpu in (("E01", "gpu-a"), ("E02", "gpu-b")):
            report = json.loads((factorial_dir / v / "train/report.json").read_text())
            assert report["gpu"] == gpu
        assert not (factorial_dir / "B0").exists()
    assert not failures, failures
    print("PASS: iSAID data gate, 5/15/40-epoch schedules, recovery, failure isolation, retention and factorial selection")


if __name__ == "__main__":
    launcher = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("run_isaid_suite.sh")
    verify(launcher.resolve())
