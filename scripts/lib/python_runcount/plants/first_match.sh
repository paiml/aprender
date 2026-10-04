#!/usr/bin/env bash
# python_runcount.sh --self-test plant: data files on the argv after the entry
# point (P25). The entry point is what Python opened FIRST that the argv names:
# the script, or the -m module, never the data file it reads next.
set -euo pipefail
python3 sub/tool.py sub/a.py
python3 -m compileall sub/b.py
