#!/usr/bin/env bash
# coverage_ratchet_target.sh - APR-RELEASE-001 P3: the nightly coverage job is report-only, target = baseline + 2.
#
# THE RULE (docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md, P3 Ratchet): "Nightly coverage job, low
# priority, report-only: target = measured baseline + 2 points per ratchet toward 95 %." Nothing enforced the
# arithmetic or the "report-only, off the PR path" half (aprender#6620).
#
# WHAT IT CHECKS, from the Makefile and the workflows, never typed here:
#   1. COV_FLOOR (the measured baseline) and COV_THRESHOLD (the 95 % end target) are integers, FLOOR <= THRESHOLD.
#   2. The next ratchet target = min(COV_THRESHOLD, COV_FLOOR + 2); printed so the report carries it.
#   3. No PR-path, merge-queue or release workflow (ci.yml, pr-gate.yml) runs coverage: it stays nightly (C324).
#
# TIER. LAB (C324): lives in scripts/lab/, runs nightly on an idle machine, never in a PR, merge-queue or release job.
# Exit 0 = ok, 1 = a rule is broken, 2 = cannot measure. `--self-test` plants each violation and proves RED.

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1


mkvar() { sed -n "s/^$1[[:space:]]*:=[[:space:]]*\([0-9][0-9]*\)[[:space:]]*\$/\1/p" "$MAKEFILE" | head -1; }

check() {
    local floor thr tgt rc=0 f hits
    MAKEFILE="${COV_MAKEFILE:-Makefile}"
    PR_WORKFLOWS="${COV_PR_WORKFLOWS:-.github/workflows/ci.yml .github/workflows/pr-gate.yml}"
    [ -r "$MAKEFILE" ] || { echo "cannot measure: $MAKEFILE unreadable"; return 2; }
    floor="$(mkvar COV_FLOOR)"; thr="$(mkvar COV_THRESHOLD)"
    if [ -z "$floor" ] || [ -z "$thr" ]; then echo "cannot measure: COV_FLOOR/COV_THRESHOLD not integers in $MAKEFILE"; return 2; fi
    if [ "$floor" -gt "$thr" ]; then echo "RED  baseline $floor > end target $thr"; rc=1; fi
    tgt=$((floor + 2)); [ "$tgt" -gt "$thr" ] && tgt="$thr"
    echo "baseline=$floor next_target=$tgt end_target=$thr (report-only)"
    for f in $PR_WORKFLOWS; do
        [ -r "$f" ] || continue
        hits="$(grep -v '^[[:space:]]*#' "$f" | grep -c 'make coverage\|cargo llvm-cov\|COV_FLOOR' || true)"
        if [ "$hits" -gt 0 ]; then echo "RED  $f runs coverage on a PR/queue/release path ($hits lines); it is nightly only"; rc=1; fi
    done
    return "$rc"
}

self_test() {
    local rc fails=0
    D="$(mktemp -d "${TMPDIR:-/tmp}/covratchet.XXXXXX")"; trap 'rm -rf "${D:?}"' EXIT
    printf 'COV_THRESHOLD := 95\nCOV_FLOOR := 89\n' >"$D/ok.mk"; printf 'name: CI\n# coverage: llvm-cov nightly\n' >"$D/ok.yml"
    COV_MAKEFILE="$D/ok.mk" COV_PR_WORKFLOWS="$D/ok.yml" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 0 ] || { echo "self-test FAIL: clean tree gave rc=$rc"; fails=1; }
    printf 'COV_THRESHOLD := 95\nCOV_FLOOR := 96\n' >"$D/bad.mk"
    COV_MAKEFILE="$D/bad.mk" COV_PR_WORKFLOWS="$D/ok.yml" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 1 ] || { echo "self-test FAIL: floor above target gave rc=$rc"; fails=1; }
    printf 'steps:\n  - run: make coverage\n' >"$D/bad.yml"
    COV_MAKEFILE="$D/ok.mk" COV_PR_WORKFLOWS="$D/bad.yml" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 1 ] || { echo "self-test FAIL: coverage on a PR path gave rc=$rc"; fails=1; }
    printf 'COV_FLOOR := 94\nCOV_THRESHOLD := 95\n' >"$D/cap.mk"
    [ "$(COV_MAKEFILE="$D/cap.mk" COV_PR_WORKFLOWS="$D/ok.yml" check | sed -n 's/.*next_target=\([0-9]*\).*/\1/p')" = 95 ] || { echo "self-test FAIL: target not capped at 95"; fails=1; }
    : >"$D/empty.mk"
    COV_MAKEFILE="$D/empty.mk" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 2 ] || { echo "self-test FAIL: unmeasurable gave rc=$rc"; fails=1; }
    [ "$fails" -eq 0 ] && echo "self-test ok: clean, floor>target, PR-path coverage, cap and not-measured all behave"
    return "$fails"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") check ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
