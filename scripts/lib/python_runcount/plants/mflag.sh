#!/usr/bin/env bash
# python_runcount.sh --self-test plant: -m and -c joined to other flags or to their value (P19).
set -euo pipefail
python3 -mcompileall -q sub
python3 -Im json.tool
python3 -Bc 'pass'
