#!/usr/bin/env bash
# check_mutants_diff_gate.sh -- run scripts/mutants_diff_gate.sh's case table (#4142) on every PR. guard-tree runs
# every tracked scripts/check_*.sh; the gate script itself is not named check_*, so without this its table is dark.
# Cargo-free: the table drives a stub in place of the mutation tool.
set -uo pipefail
case "${1:-}" in -h|--help) echo "usage: bash scripts/check_mutants_diff_gate.sh"; exit 0 ;; esac
exec bash "$(dirname "$0")/mutants_diff_gate.sh" --self-test
