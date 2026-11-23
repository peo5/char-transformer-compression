#!/usr/bin/env bash
# Rebuild the generated part of RESULTS.md (everything after the marker) from results/, using src/tables.py.
# Run from the repository root.
set -e
sed -n '1,/<!-- tables.py output below -->/p' RESULTS.md > RESULTS.tmp
python3 src/tables.py results >> RESULTS.tmp
mv RESULTS.tmp RESULTS.md
