#!/usr/bin/env bash
# One-time setup: a Python 3.11 venv in .venv with torch, numpy, pyarrow and matplotlib. Needs uv
# (https://docs.astral.sh/uv/). The torch index below is the CUDA 12.6 build; pick the one for your driver.
set -e
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy pyarrow matplotlib
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch, numpy, pyarrow; print(torch.__version__, torch.cuda.is_available())"
