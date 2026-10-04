#!/usr/bin/env bash
# python_runcount.sh --self-test plant: argv[0] renamed on an unshimmed interpreter (P14).
set -euo pipefail
( exec -a renamed python3.11 sub/tool.py )
