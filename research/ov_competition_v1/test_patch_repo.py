from pathlib import Path
import subprocess
import sys

from .patch_repo import ANCHOR, REPLACEMENT


def test_adapter_patch_fail_closed(tmp_path):
    p = tmp_path / 'model.py'
    p.write_text('class Mock:\n    def forward(self):\n'+ANCHOR)
    original = p.read_bytes()
    tool = Path(__file__).with_name('patch_repo.py')
    a = subprocess.run([sys.executable,str(tool),'--target',str(p),'--check'],
                       capture_output=True,text=True)
    assert a.returncode == 0, a.stderr
    assert p.read_bytes() == original  # Read-only check must not modify.
    b = subprocess.run([sys.executable,str(tool),'--target',str(p),'--apply'],
                       capture_output=True,text=True)
    assert b.returncode == 0, b.stderr
    patched = p.read_text()
    assert REPLACEMENT in patched
    assert ANCHOR not in patched
    assert 'for layer_index, layer in enumerate(self.layers)' in patched
    c = subprocess.run([sys.executable,str(tool),'--target',str(p),'--apply'],
                       capture_output=True,text=True)
    assert c.returncode != 0  # No silent duplicate injection.


def test_adapter_patch_rejects_unknown_version(tmp_path):
    p = tmp_path / 'model.py'
    p.write_text('def forward():\n    pass\n')
    tool = Path(__file__).with_name('patch_repo.py')
    p0 = p.read_bytes()
    r = subprocess.run([sys.executable,str(tool),'--target',str(p),'--apply'],
                       capture_output=True,text=True)
    assert r.returncode != 0
    assert p.read_bytes() == p0
