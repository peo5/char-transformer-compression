#!/usr/bin/env bash
# Step 2b: general-purpose compressors on the normalized test files, in bits per character.
# Usage: bash compressors.sh DATA_DIR > compressors.tsv
echo -e "lang\tcompressor\tchars\tbytes\tbits_per_char"
for lang in pt en; do
  f=$1/$lang/test.txt
  n=$(stat -c %s "$f")
  for c in "gzip -9" "xz -9e" "zstd -19 --long=27"; do
    b=$($c -c < "$f" | wc -c)
    echo -e "$lang\t$c\t$n\t$b\t$(echo "scale=4; 8*$b/$n" | bc)"
  done
done
