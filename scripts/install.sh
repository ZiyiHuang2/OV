#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pip install 'pip>=24' 'setuptools<81' wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/test.txt
# CAT-Seg uses PyTorch tensor operations, not Detectron2 CUDA ROI/NMS ops.
# Build the required Detectron2 extension on CPU: nvcc is not required.
CC="${CC:-gcc}" CXX="${CXX:-g++}" CUDA_VISIBLE_DEVICES="" MAX_JOBS="${MAX_JOBS:-2}" python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298'
python -c 'import torch, detectron2, timm; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "detectron2", detectron2.__version__, "timm", timm.__version__)'
