#!/usr/bin/env bash
# check_tool_versions_test.sh — falsifier for scripts/check_tool_versions.sh
# (BSE-10a, PMAT-1066).
#
# Three things, each independently proven:
#   1. the guard's own --self-test case table is green (right/wrong/absent
#      version on PATH, both tools);
#   2. --audit-workflow against the REAL .github/workflows/ci.yml is clean
#      (no `cargo install pmat|bashrs`, no fail-open invocation of this
#      guard);
#   3. a MUTANT copy of ci.yml with an install line re-added goes RED under
#      the same audit -- proving the audit is load-bearing rather than
#      vacuously green on any input (the capability-gated-tests-are-vacuous
#      lesson: a falsifier that cannot fail proves nothing).
#
#   bash scripts/tests/check_tool_versions_test.sh
#
# Refs: PMAT-1066, BSE-10a.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD="$ROOT/scripts/check_tool_versions.sh"
WORKFLOW="$ROOT/.github/workflows/ci.yml"

if [ ! -f "$GUARD" ]; then
    printf 'ENV - %s is missing (the box cannot answer)\n' "$GUARD" >&2
    exit 2
fi
if [ ! -f "$WORKFLOW" ]; then
    printf 'ENV - %s is missing (the box cannot answer)\n' "$WORKFLOW" >&2
    exit 2
fi

n=0
red=0

row() { # row WANT_RC LABEL MUST_MATCH -- CMD...
    local want=$1 label=$2 pat=$3 rc=0 out
    shift 3
    n=$((n + 1))
    out=$("$@" 2>&1); rc=$?
    if [ "$rc" = "$want" ] && printf '%s' "$out" | grep -qE -- "$pat"; then
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else
        printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
        printf '%s\n' "$out" | sed 's/^/        /'
        red=1
    fi
}

# ---------------------------------------------------------------------------
# 1. The guard's own case table.
# ---------------------------------------------------------------------------
row 0 "check_tool_versions.sh --self-test is itself green" \
    'SELF-TEST PASSED' bash "$GUARD" --self-test

# ---------------------------------------------------------------------------
# 2. The real workflow is clean.
# ---------------------------------------------------------------------------
row 0 "--audit-workflow against the real ci.yml is clean" \
    '^PASS  .*no cargo install of pmat/bashrs' bash "$GUARD" --audit-workflow "$WORKFLOW"

# ---------------------------------------------------------------------------
# 3. Mutants: re-add each forbidden shape to a TEMP COPY of ci.yml and prove
# the audit turns RED on each independently. Never mutate the tracked file.
# ---------------------------------------------------------------------------
TD=$(mktemp -d "${TMPDIR:-/tmp}/tool-versions-audit-test.XXXXXX") || exit 2
cleanup() {
    local victim=${TD:-}
    case "$victim" in
        *tool-versions-audit-test.*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
        *) return 0 ;;
    esac
}
trap cleanup EXIT

cp "$WORKFLOW" "$TD/reinstall.yml"
printf '\n      - name: mutant re-add\n        run: cargo install bashrs --locked --quiet || true\n' >> "$TD/reinstall.yml"
row 1 "mutant: a re-added \`cargo install bashrs\` line goes RED" \
    'FAIL.*cargo install.*pmat/bashrs' bash "$GUARD" --audit-workflow "$TD/reinstall.yml"

cp "$WORKFLOW" "$TD/failopen.yml"
printf '\n      - name: mutant fail-open\n        run: bash scripts/check_tool_versions.sh || true\n' >> "$TD/failopen.yml"
row 1 "mutant: a fail-open \`check_tool_versions.sh || true\` call goes RED" \
    'FAIL.*fail-open' bash "$GUARD" --audit-workflow "$TD/failopen.yml"

# A missing file is ENV (exit 2), never a pass.
row 2 "a missing workflow file is ENV (exit 2), never a pass" \
    'does not exist' bash "$GUARD" --audit-workflow "$TD/absent.yml"

printf '%s/%s checks, %s failed\n' "$((n - red))" "$n" "$red"
[ "$red" = 0 ]
