#!/usr/bin/env bash
# scripts/lab/reconcile_lab_scope.sh -- #6605 (spec sweep, release-reconcile.md R-c). Nightly check (C324).
# The spec line: R-c (scripts/check_reconcile.sh) is a release gate since the operator's yes (RECON1 2026-11-17, #7194),
# never in a PR or merge-queue job. This check holds that to account:
#   1. spec    release-reconcile.md says R-c is a release gate (RECON1) and names "never in a PR".
#   2. script  scripts/check_reconcile.sh exists.
#   3. scope   no workflow runs check_reconcile.sh unless the `on:` block of that workflow has only schedule,
#              workflow_dispatch or workflow_call triggers (no pull_request, pull_request_target, merge_group, push or
#              release). Only the `on:` block is read: a job keyed `release:` is not a trigger. workflow_call is allowed
#              because its callers (release-rehearsal-nightly) are scheduled.
# Usage: scripts/lab/reconcile_lab_scope.sh | --self-test      Exit: 0 green, 1 RED, 2 cannot measure
# Overrides (tests): SPEC SCRIPT WORKFLOWS_DIR
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPEC="${SPEC:-$ROOT/docs/specifications/release-reconcile.md}"
SCRIPT="${SCRIPT:-$ROOT/scripts/check_reconcile.sh}"
WORKFLOWS_DIR="${WORKFLOWS_DIR:-$ROOT/.github/workflows}"

# on_block FILE: the lines of the top-level `on:` mapping (up to the next top-level key).
on_block() { awk '/^"?on"?:/ { on = 1; next } on && /^[^[:space:]#]/ { on = 0 } on { print }' "$1"; }

run() {
    local red=0 f bad="" n=0
    for f in "$SPEC" "$SCRIPT"; do [ -f "$f" ] || { echo "NO-MEASURE: $f does not exist" >&2; return 2; }; done
    [ -d "$WORKFLOWS_DIR" ] || { echo "NO-MEASURE: $WORKFLOWS_DIR does not exist" >&2; return 2; }
    if grep -Eq 'R-c .*is a \*\*release gate\*\*' "$SPEC" && grep -Eq 'RECON1' "$SPEC" && grep -Eq 'never in a PR' "$SPEC"; then echo "  ok    spec   R-c is a release gate (RECON1), never in a PR"
    else echo "  RED   spec   release-reconcile.md no longer says R-c is a release gate (RECON1) / never in a PR"; red=1; fi
    for f in "$WORKFLOWS_DIR"/*.yml "$WORKFLOWS_DIR"/*.yaml; do
        [ -f "$f" ] || continue
        grep -q 'check_reconcile' "$f" || continue
        n=$((n+1))
        if on_block "$f" | grep -Eq '^[[:space:]]+(pull_request|pull_request_target|merge_group|push|release):?'; then
            bad="$bad $(basename "$f")"
        fi
    done
    if [ -z "$bad" ]; then echo "  ok    scope  $n workflow(s) run check_reconcile.sh; none on a PR/merge-queue/push/release trigger"
    else echo "  RED   scope  check_reconcile.sh runs on a non-LAB trigger in:$bad"; red=1; fi
    if [ "$red" -eq 0 ]; then echo "GREEN reconcile LAB scope"; else echo "RED reconcile LAB scope"; fi
    return "$red"
}

self_test() {
    local d rc fail=0
    d="$(mktemp -d /tmp/reclab.XXXXXX)"
    mkdir -p "$d/wf"; cp "$SPEC" "$d/spec.md"; cp "$SCRIPT" "$d/s.sh"
    expect() { # name want-rc
        set +e; SPEC="$d/spec.md" SCRIPT="$d/s.sh" WORKFLOWS_DIR="$d/wf" "$0" >/dev/null 2>&1; rc=$?; set -e
        if [ "$rc" = "$2" ]; then echo "  pass  $1 (rc=$rc)"; else echo "  FAIL  $1: want $2 got $rc"; fail=1; fi
    }
    expect "real spec + no reconcile workflow is GREEN" 0
    printf 'on:\n  workflow_dispatch:\njobs:\n  release:\n    steps:\n      - run: scripts/check_reconcile.sh\n' > "$d/wf/rel.yml"; expect "job keyed release: under dispatch is GREEN" 0
    printf 'on:\n  workflow_call:\n  schedule:\n    - cron: "0 3 * * *"\njobs:\n  r:\n    steps:\n      - run: scripts/check_reconcile.sh\n' > "$d/wf/rel.yml"; expect "workflow_call + schedule is GREEN" 0
    printf 'on:\n  release:\n    types: [published]\njobs:\n  r:\n    steps:\n      - run: scripts/check_reconcile.sh\n' > "$d/wf/rel.yml"; expect "release: trigger is RED" 1
    rm -f "${d:?}/wf/rel.yml"
    printf 'on:\n  schedule:\n    - cron: "0 3 * * *"\njobs:\n  r:\n    steps:\n      - run: scripts/check_reconcile.sh\n' > "$d/wf/nightly.yml"; expect "nightly-only workflow is GREEN" 0
    printf 'on:\n  pull_request:\n    branches: [main]\njobs:\n  r:\n    steps:\n      - run: scripts/check_reconcile.sh\n' > "$d/wf/ci.yml"; expect "reconcile on pull_request is RED" 1
    printf 'on:\n  merge_group:\njobs:\n  r:\n    steps:\n      - run: scripts/check_reconcile.sh\n' > "$d/wf/ci.yml"; expect "reconcile on merge_group is RED" 1
    printf 'on:\n  push:\n    tags: ["v*"]\njobs:\n  r:\n    steps:\n      - run: scripts/check_reconcile.sh\n' > "$d/wf/ci.yml"; expect "reconcile on release push is RED" 1
    rm -f "${d:?}/wf/ci.yml"; sed -i 's/R-c (`scripts\/check_reconcile.sh`) is a \*\*release gate\*\*/R-c is a lane/' "$d/spec.md"; expect "spec dropped release gate is RED" 1
    : > "$d/spec.md"; expect "empty spec is RED (no LAB statement)" 1
    mv "$d/s.sh" "$d/s.gone"; expect "missing script is NO-MEASURE" 2
    rm -rf "${d:?}"
    return "$fail"
}

if [ "${1:-}" = "--self-test" ]; then self_test; else run; fi
