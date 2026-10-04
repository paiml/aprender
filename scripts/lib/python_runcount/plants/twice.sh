#!/usr/bin/env bash
# python_runcount.sh --self-test plant: the same script run twice is one entry point (P11).
set -euo pipefail
python3 sub/tool.py
python3 sub/tool.py
