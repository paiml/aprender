#!/usr/bin/env bash
# unsloth_ft_apr_side.sh — apr side runner for unsloth_finetune_throughput.sh.
#
# Runs unsloth_ft_apr.py (an adapter around `apr finetune`). All arguments pass through.
# APR_BIN names the apr binary (default: apr on PATH); build it BEFORE taking the GPU
# lock, never under it. Exit 4 = apr's receipt lacks a key the verdict needs (an R15 gap).
set -uo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
exec python3 "$root/scripts/bench/unsloth_ft_apr.py" "$@"
