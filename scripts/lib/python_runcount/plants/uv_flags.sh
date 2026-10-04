#!/usr/bin/env bash
# python_runcount.sh --self-test plant: uv options that take a value, before the script (P21).
set -euo pipefail
uv run --color never --index-strategy unsafe-best-match -c cons.txt sub/tool.py
