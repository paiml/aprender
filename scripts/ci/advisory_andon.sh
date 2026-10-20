#!/usr/bin/env bash
# advisory_andon.sh -- main-resident advisories are an andon plus a P0 ticket, never a PR block
# (PMAT-5792, contracts/advisory-gate-diff-v1.yaml main_resident_advisory). Runs ONLY on a
# schedule or push to main, never on a PR or merge_group path (C324: no third state there).
#
#   advisory_andon.sh keys FILE      finding keys of a cargo-deny NDJSON file (advisory_diff.sh keys)
#   advisory_andon.sh due CREATED NOW   exit 0 if the ticket is older than 4h (label p0-overdue), else 1
#   advisory_andon.sh --self-test
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUE_SECS=$((4 * 3600))

due() { # ISO created, ISO now
    local c n
    c=$(date -u -d "$1" +%s) n=$(date -u -d "$2" +%s)
    [ $((n - c)) -gt "$DUE_SECS" ]
}

self_test() {
    local bad=0 rc
    chk() { rc=0; due "$2" "$3" || rc=$?; [ "$rc" -eq "$4" ] || { echo "SELF-TEST FAIL: $1 want $4 got $rc" >&2; bad=1; }; }
    chk 'under 4h is not due' 2026-10-20T10:00:00Z 2026-10-20T13:59:00Z 1
    chk 'exactly 4h is not due' 2026-10-20T10:00:00Z 2026-10-20T14:00:00Z 1
    chk 'over 4h is due' 2026-10-20T10:00:00Z 2026-10-20T14:00:01Z 0
    [ "$bad" -eq 0 ] && echo "advisory_andon self-test: 3 cases ok"
    return "$bad"
}

case "${1:-}" in
    keys) bash "$HERE/advisory_diff.sh" keys "$2" ;;
    due) due "$2" "$3" ;;
    --self-test) self_test ;;
    *) echo "usage: $0 keys FILE | due CREATED NOW | --self-test" >&2; exit 2 ;;
esac
