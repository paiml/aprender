#!/usr/bin/env bash
# check_release_wall_time.sh -- the case table for scripts/release/release_wall_time.py (#4045 M7): release-night wall
# time from the systems of record, every mutant killed by its named row. Hermetic (injected anchors, no network),
# cargo-free, so guard_tree runs it on every PR.
set -euo pipefail
case "${1:-}" in -h|--help) echo "usage: bash scripts/check_release_wall_time.sh   (the case table; no arguments)"; exit 0 ;; esac
cd "$(dirname "$0")/.." || exit 2
rc=0; out=$(python3 scripts/release/release_wall_time_cases.py --mutants 2>&1) || rc=$?
printf '%s\n' "$out" | grep -vE '^PASS' || true   # every row PASSing prints nothing here
echo "check_release_wall_time: $([ "$rc" = 0 ] && echo PASS || echo FAIL)"
exit "$rc"
