#!/usr/bin/env bash
# Rebuild the generated part of RESULTS.md (everything after the marker) from results/ with tables.py.
set -e
sed -n '1,/<!-- tables.py output below -->/p' RESULTS.md > RESULTS.tmp
python tables.py results >> RESULTS.tmp
mv RESULTS.tmp RESULTS.md
