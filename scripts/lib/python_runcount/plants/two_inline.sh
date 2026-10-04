#!/usr/bin/env bash
# python_runcount.sh --self-test plant: two different -c snippets and a heredoc in
# one lane (N2). Inline code is keyed by the lane that holds it: one entry point.
set -euo pipefail
python3 -c 'pass'
python3 -c 'print(2)'
python3 - <<'PYCODE'
print(3)
PYCODE
