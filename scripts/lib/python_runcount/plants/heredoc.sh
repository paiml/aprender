#!/usr/bin/env bash
# python_runcount.sh --self-test plant: Python code on stdin from a heredoc (P3).
set -euo pipefail
python3 - <<'PYCODE'
print("planted")
PYCODE
