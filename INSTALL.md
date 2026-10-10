# Installation

The supported OVRS entry point uses Linux, Python 3.10–3.12, OpenAI CLIP
ViT-B/16, PyTorch 2.5.1 and a pinned Detectron2 source revision.

```bash
conda create -n ovrs python=3.10 -y
conda activate ovrs
bash scripts/install.sh
python -m pytest -q
```

Install git and gcc/g++ first. The script installs CUDA 12.1 PyTorch wheels,
and builds Detectron2 CPU extensions without nvcc. This CAT-Seg training path
uses GPU PyTorch operations, not Detectron2 CUDA ROI/NMS operations.

`requirements.txt` installs the common runtime packages; PyTorch and
Detectron2 must also be installed. Prefer the complete script in a new
environment. Data preparation, synthetic smoke training, ablation commands
and checkpoint evaluation are documented in [README.md](README.md).

Actual pre-release checks and GPU limitations are in
[VALIDATION.md](VALIDATION.md).
