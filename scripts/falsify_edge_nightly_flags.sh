#!/usr/bin/env bash
# Flag case table of scripts/edge_fast_verbs_nightly.sh (#7156): flags are parsed before any tool or model is
# touched, so a bad value dies with its own message and a good one reaches the next check. Exit 0 held, 1 broke.
set -uo pipefail
S="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/edge_fast_verbs_nightly.sh"
bad=0
row() { # row "ARGS" "message fragment"; every row exits 2
    local out rc
    out=$(bash "$S" $1 2>&1); rc=$?
    [ "$rc" -eq 2 ] && [[ "$out" == *"$2"* ]] && return 0
    printf 'FAIL  %s (rc %s, wanted 2 with "%s"): %s\n' "$1" "$rc" "$2" "$out"; bad=1
}
for n in 0 1 2 3 4 5; do row "--only $n" "--model is required"; done
for v in 6 -1 01 x; do row "--only $v" "--only must be one of 0..5"; done
for v in 0 abc -5; do row "--step-timeout $v" "--step-timeout must be a positive"; done
row "--only" "--only needs a value"; row "--step-timeout" "--step-timeout needs a value"
row "--only 2 --step-timeout 30" "--model is required"
exit "$bad"
