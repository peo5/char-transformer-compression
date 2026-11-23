#!/usr/bin/env bash
# One-time setup, from the repository root. It creates a Python 3.11 environment in .venv with torch, numpy,
# pyarrow and matplotlib. It needs uv (https://docs.astral.sh/uv/). The torch index below is the build for
# CUDA 12.6; choose the one that matches your GPU driver.
set -e
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy pyarrow matplotlib
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch, numpy, pyarrow; print(torch.__version__, torch.cuda.is_available())"
