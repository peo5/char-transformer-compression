#!/usr/bin/env bash
# All GPU work, in order of priority. Re-running skips what is already done.
# Usage (in the working directory, after setup.sh and prepare_data.py): setsid -f nice -n 19 bash run_gpu.sh > run_gpu.log 2>&1 < /dev/null
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8          # needed by torch.use_deterministic_algorithms in coder.py
PY=.venv/bin/python
mkdir -p runs/coder
for size in 0.1M 1M 10M; do
  for lang in pt en; do
    [ -f runs/${lang}_${size}.pt ] || $PY train.py $lang $size data runs
  done
  $PY evaluate.py data runs                       # skips pairs already in runs/eval/summary.json
  for lang in pt en; do                           # bit-exact round trip on the first 1 MB of the test set
    [ -f runs/coder/${lang}_${size}.json ] || $PY coder.py runs/${lang}_${size}.pt $size data/$lang/test.txt 1000000 runs/coder/${lang}_${size}
  done
  $PY lid.py runs runs/lid.json
done
echo ALL DONE
