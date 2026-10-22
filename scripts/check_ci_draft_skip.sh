#!/usr/bin/env bash
# check_ci_draft_skip.sh -- a draft PR does not run the full CI, and the required checks still report.
#
# THE RULE (operator, via arbiter-cop, 2026-10-21 ~13:52Z: "yes to all three", item 3). On a draft pull_request:
#   1. `on.pull_request.types` is [opened, synchronize, reopened, ready_for_review], so marking a PR ready fires the full run.
#   2. the heavy jobs x86-main and workspace-test-shard carry `github.event.pull_request.draft != true` in their `if:`.
#   3. the three verdict jobs (workspace-test, ci-gate, gate) read `DRAFT: ${{ github.event.pull_request.draft }}` and
#      exit 0 on `true`, so `workspace-test`, `ci / gate` and `gate` still REPORT (a draft cannot be queued or merged).
#   4. workspace-test's checkout / download-artifact / sigma steps are skipped on a draft (they would fail with no shards).
# No third state: nothing here is continue-on-error; a non-draft PR, a merge_group entry and a push run exactly as before.
#
# Usage:  bash scripts/check_ci_draft_skip.sh [--self-test] [WORKFLOW_FILE]
# Wired:  ci/explicit-test-commands.d/840-ci-draft-skip.cmd (--self-test) and guard_tree.sh (no args).
set -euo pipefail

# job_block FILE JOB -> the lines of `  JOB:` up to the next two-space-indented key
job_block() {
    awk -v j="$2" '
        $0 ~ "^  " j ":[[:space:]]*$" { on = 1; print; next }
        on && /^  [A-Za-z0-9_-]+:/ { exit }
        on { print }' "$1"
}

check() { # FILE -> prints one FAIL line per violated condition; rc 1 when any
    local f="$1" bad=0 j blk
    if ! awk '/^  pull_request:/{p=1;next} p&&/^  [a-z_]+:/{exit} p&&/types:/{print}' "$f" \
        | /usr/bin/grep -q 'opened.*synchronize.*reopened.*ready_for_review'; then
        echo "FAIL  on.pull_request.types lacks opened, synchronize, reopened, ready_for_review"; bad=1
    fi
    for j in x86-main workspace-test-shard; do
        blk="$(job_block "$f" "$j")"
        if [ -z "$blk" ]; then echo "FAIL  job $j not found"; bad=1; continue; fi
        if ! printf '%s\n' "$blk" | /usr/bin/grep -E '^    if:' | /usr/bin/grep -q 'github.event.pull_request.draft != true'; then
            echo "FAIL  job $j has no draft guard in its if:"; bad=1
        fi
    done
    for j in workspace-test ci-gate gate; do
        blk="$(job_block "$f" "$j")"
        if [ -z "$blk" ]; then echo "FAIL  job $j not found"; bad=1; continue; fi
        if ! printf '%s\n' "$blk" | /usr/bin/grep -q 'DRAFT: ${{ github.event.pull_request.draft }}'; then
            echo "FAIL  verdict job $j does not read the draft flag"; bad=1
        fi
        if ! printf '%s\n' "$blk" | /usr/bin/grep -q '\[ "${DRAFT:-}" != true \] || {.*exit 0'; then
            echo "FAIL  verdict job $j does not report green on a draft"; bad=1
        fi
        if printf '%s\n' "$blk" | /usr/bin/grep -q 'continue-on-error'; then
            echo "FAIL  verdict job $j uses continue-on-error (no third state)"; bad=1
        fi
    done
    blk="$(job_block "$f" workspace-test)"
    if [ "$(printf '%s\n' "$blk" | /usr/bin/grep -c 'if: ${{ github.event.pull_request.draft != true }}')" -lt 3 ]; then
        echo "FAIL  workspace-test: checkout, download-artifact and sigma steps are not all skipped on a draft"; bad=1
    fi
    return "$bad"
}

self_test() {
    local src="$1" td fails=0 got name
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    if got="$(check "$src")"; then echo "ok    green the workflow as committed"; else echo "FAIL  green: the workflow as committed is RED: $got"; fails=1; fi
    mutant() { # NAME SED_SCRIPT -> the mutated workflow must be RED
        name="$1"; sed -E "$2" "$src" > "$td/m.yml"
        if cmp -s "$src" "$td/m.yml"; then echo "FAIL  mutant $name changed nothing"; fails=1; return; fi
        if got="$(check "$td/m.yml")"; then echo "FAIL  mutant $name stayed GREEN"; fails=1; else echo "ok    red   mutant $name"; fi
    }
    mutant no-ready-for-review 's/, ready_for_review\]/]/'
    mutant x86-main-unguarded 's/ && github\.event\.pull_request\.draft != true \}\}/ }}/'
    mutant shard-unguarded '/^  workspace-test-shard:$/,/runs-on:/{/draft != true/d;}'
    mutant gate-no-draft-exit '/echo "gate: draft PR/d'
    mutant ci-gate-no-draft-flag '/^  ci-gate:/,/^  gate:/{/DRAFT: \$/d}'
    mutant sigma-step-runs-on-draft '/Σ-executed/{n;d}'
    mutant third-state 's/^  gate:$/  gate:\n    continue-on-error: true/'
    return "$fails"
}

root="$(git rev-parse --show-toplevel)"
case "${1:-}" in
    --self-test) self_test "${2:-$root/.github/workflows/ci.yml}"; exit $? ;;
    *)
        out="$(check "${1:-$root/.github/workflows/ci.yml}")" && { echo "ok    drafts skip the heavy jobs and the required checks still report"; exit 0; }
        printf '%s\n' "$out"; exit 1 ;;
esac
