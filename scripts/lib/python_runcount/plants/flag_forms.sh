#!/usr/bin/env bash
# python_runcount.sh --self-test plant: joined -W/-X values and a combined bash -ec (P18).
set -euo pipefail
python3 -Wonce sub/tool.py
python3 -Xutf8 -c 'pass'
bash -ec "python3 -c 'pass'; :"
