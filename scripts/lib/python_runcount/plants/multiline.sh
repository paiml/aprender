#!/usr/bin/env bash
# python_runcount.sh --self-test plant: inline code with a newline and a tab (P17).
# The shim log is one record per line, tab separated, so the code must not split it.
set -euo pipefail
python3 -c "$(printf 'import json\nif 1:\n\tprint(json.dumps({}, indent=2))')"
