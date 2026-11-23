#!/usr/bin/env bash
# Optional sanity check data: text8 (first 1e8 chars of enwik9, cleaned to a-z and space), standard split
# 90M train / 5M val / 5M test. The test part is stored as val.txt so train.py reports test bpc at the end.
# Usage: bash text8_data.sh OUT_DIR    -> OUT_DIR/text8/{train,val}.txt
set -e
mkdir -p $1/text8 && cd $1/text8
curl -sSL -o text8.zip http://mattmahoney.net/dc/text8.zip
unzip -o -q text8.zip && rm text8.zip
head -c 90000000 text8 > train.txt
tail -c 5000000 text8 > val.txt
rm text8
