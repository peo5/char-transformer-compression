#!/usr/bin/env bash
# Optional sanity check data: text8, the first 10^8 characters of enwik9 (English Wikipedia), cleaned to a-z
# and space, with the standard split of 90M train / 5M validation / 5M test characters. The test part is saved
# as val.txt, so train.py reports the test bits per character at the end.
# Usage: bash src/text8_data.sh OUT_DIR    -> OUT_DIR/text8/{train,val}.txt
set -e
mkdir -p $1/text8 && cd $1/text8
curl -sSL -o text8.zip http://mattmahoney.net/dc/text8.zip
unzip -o -q text8.zip && rm text8.zip
head -c 90000000 text8 > train.txt
tail -c 5000000 text8 > val.txt
rm text8
