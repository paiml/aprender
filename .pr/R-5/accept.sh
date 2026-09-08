#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
red=0; n=0
leg() { n=$((n+1)); if "$@" >/tmp/r5-leg.$n 2>&1; then echo "ok    A$n  $*"; else echo "FAIL  A$n  $* (rc=$?)"; tail -3 /tmp/r5-leg.$n; red=1; fi; }
leg bash -c 'out=$(bash scripts/publish_cascade.sh --help 2>&1); case "$out" in *--dry-run*) exit 0 ;; *) exit 1 ;; esac'
leg bash scripts/publish_cascade.sh --self-test
leg bash -c '. scripts/pv_bin.sh >/dev/null 2>&1 && "$PV" validate contracts/apr-publish-cascade-v1.yaml'
leg bash -c 'out=$(bash scripts/publish_cascade.sh --list 2>&1); [ -n "$out" ]'
# NOT `bashrs lint … | grep -q "0 error"`: `0 error` is a substring of `10 error(s)`, which is
# the pass-grep class check_pass_grep_anchored.sh exists to refuse, and `producer | grep -q`
# SIGPIPEs the producer under pipefail. Capture, then count an ANCHORED pattern.
leg bash -c 'out=$(bashrs lint scripts/publish_cascade.sh 2>&1); [ "$(printf "%s" "$out" | grep -c "\\[error\\]")" = 0 ]'
echo "$n legs"; [ "$red" = 0 ]
