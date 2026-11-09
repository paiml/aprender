#!/usr/bin/env bash
# report_only_bench.sh - beat-campaign schedule: PMAT-728-GPU (Qwen3-370M Blackwell baseline) and PMAT-715 (throughput
# vs Unsloth) are REPORT-ONLY (aprender#6618, #6619; docs/specifications/beat-campaign-10day-schedule-2026-06-12.md).
#
# WHAT IT CHECKS: (1) the spec still marks both lines REPORT-ONLY; (2) no PR-path, merge-queue or release workflow
# (ci.yml, pr-gate.yml) runs either bench (a non-comment line naming PMAT-728, PMAT-715 or unsloth), so neither can block.
# TIER. LAB (C324): scripts/lab/, nightly, never in a PR, merge-queue or release job.
# Exit 0 ok, 1 broken, 2 cannot measure. `--self-test` plants each violation.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

check() {
    local spec wf f rc=0 id hits
    spec="${RO_SPEC:-docs/specifications/beat-campaign-10day-schedule-2026-06-12.md}"
    wf="${RO_PR_WORKFLOWS:-.github/workflows/ci.yml .github/workflows/pr-gate.yml}"
    [ -r "$spec" ] || { echo "cannot measure: $spec unreadable"; return 2; }
    for id in PMAT-728 PMAT-715; do
        if grep -E "$id" "$spec" | grep -q 'REPORT-ONLY'; then echo "ok   $id marked REPORT-ONLY in spec"
        else echo "RED  $id not marked REPORT-ONLY in $spec"; rc=1; fi
    done
    for f in $wf; do
        [ -r "$f" ] || continue
        hits="$(grep -v '^[[:space:]]*#' "$f" | grep -ci 'PMAT-728\|PMAT-715\|unsloth' || true)"
        if [ "$hits" -gt 0 ]; then echo "RED  $f runs a report-only bench on a PR/queue/release path ($hits lines)"; rc=1; fi
    done
    return "$rc"
}

self_test() {
    local rc fails=0
    D="$(mktemp -d "${TMPDIR:-/tmp}/roonly.XXXXXX")"; trap 'rm -rf "${D:?}"' EXIT
    printf -- '- PMAT-728-GPU baseline, REPORT-ONLY\n- PMAT-715 vs Unsloth, REPORT-ONLY\n' >"$D/ok.md"; printf 'name: CI\n# unsloth nightly\n' >"$D/ok.yml"
    RO_SPEC="$D/ok.md" RO_PR_WORKFLOWS="$D/ok.yml" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 0 ] || { echo "self-test FAIL: clean gave rc=$rc"; fails=1; }
    printf -- '- PMAT-728-GPU baseline, REPORT-ONLY\n- PMAT-715 vs Unsloth, blocking\n' >"$D/bad.md"
    RO_SPEC="$D/bad.md" RO_PR_WORKFLOWS="$D/ok.yml" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 1 ] || { echo "self-test FAIL: spec drift gave rc=$rc"; fails=1; }
    printf 'steps:\n  - run: bench PMAT-715\n' >"$D/bad.yml"
    RO_SPEC="$D/ok.md" RO_PR_WORKFLOWS="$D/bad.yml" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 1 ] || { echo "self-test FAIL: bench on PR path gave rc=$rc"; fails=1; }
    RO_SPEC="$D/missing.md" check >/dev/null && rc=0 || rc=$?
    [ "$rc" -eq 2 ] || { echo "self-test FAIL: unmeasurable gave rc=$rc"; fails=1; }
    [ "$fails" -eq 0 ] && echo "self-test ok: clean, spec drift, PR-path bench and not-measured all behave"
    return "$fails"
}

[ "${1:-}" = "--self-test" ] && { self_test; exit $?; }
check
