#!/usr/bin/env bash
# python_runcount.sh --self-test plant: -W/-X values joined and separated, and a
# combined bash -ec (P18). The inline code under bash -ec belongs to this lane.
set -euo pipefail
python3 -Wonce sub/tool.py
python3 -W once sub/tool.py
python3 -Xutf8 -c 'pass'
python3 -X utf8 -c 'pass'
bash -ec "python3 -c 'pass'; :"
