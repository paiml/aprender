#!/usr/bin/env bash
# python_runcount.sh --self-test plant: a script written to a fresh temp dir (P15).
set -euo pipefail
d=$(mktemp -d)
: > "$d/cell.py"
python3 "$d/cell.py"
rm -rf "${d:?}"
