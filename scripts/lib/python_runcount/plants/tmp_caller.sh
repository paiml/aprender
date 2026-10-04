#!/usr/bin/env bash
# python_runcount.sh --self-test plant: inline Python held by a mktemp script (P16).
set -euo pipefail
s=$(mktemp)
printf 'python3 -c pass\n' > "$s"
bash "$s"
rm -f "${s:?}"
