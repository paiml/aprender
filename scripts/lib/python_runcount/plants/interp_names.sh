#!/usr/bin/env bash
# python_runcount.sh --self-test plant: interpreter names past python3.NN, and env -u (P23).
set -euo pipefail
pythonw sub/a.py
pypy3 sub/b.py
python3.12d sub/c.py
env -u HOME python3 sub/d.py
