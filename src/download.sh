#!/usr/bin/env bash
# Download the Wikipedia dumps of 2023-11-01 (parquet files) into raw/. Run from the repository root.
# It takes all 6 Portuguese files and 6 of the 41 English files, spread evenly.
base=https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main
files="$(for i in 0 1 2 3 4 5; do echo 20231101.pt/train-0000$i-of-00006.parquet; done)
$(for i in 00000 00008 00016 00024 00032 00040; do echo 20231101.en/train-$i-of-00041.parquet; done)"
for f in $files; do
  mkdir -p raw/$(dirname $f)
  curl -sSL --retry 5 -C - -o raw/$f $base/$f && echo "done $f" &
done
wait
