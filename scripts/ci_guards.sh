#!/usr/bin/env bash
# ci_guards.sh -- run CI's guard steps, ALL of them, in CI and locally (#4415).
#
# ONE entry point (operator amendment A4, 2026-09-25): CI's guard-cargo and
# guard-tree jobs each run `bash scripts/ci_guards.sh <job>` as a single step,
# and `make guards-local` and the pre-push hook run this same file. The steps
# are the `<job>-steps` manifest jobs in ci/sections.yml (`if: false`, never run by
# GitHub), executed by scripts/lib/ci_guard_steps.sh (bash + awk + jq). Both places print
#   ci_guards: sha256 <runner sha> manifest <steps sha> (<jobs>)
# so "CI and my machine ran the same guards" is a line comparison.
#
# GitHub stops a job at its first red step; this does not. Every step runs,
# every failure gets an ::error annotation, and one SUMMARY lists them all.
# Off-CI, a step that cannot run here prints SKIP with the reason; a run in
# which nothing executed exits 2, never 0.
#
#   bash scripts/ci_guards.sh                        # every guard section the gate reads
#   bash scripts/ci_guards.sh guard-tree             # one job
#   bash scripts/ci_guards.sh --only 'bashrs|pinned' guard-cargo
#   bash scripts/ci_guards.sh --step-timeout 600     # seconds per step (default 4800)
#   bash scripts/ci_guards.sh --list | --sha | --check-coverage
#
# Exit: 0 all ran steps passed, 1 a step failed or timed out, 2 usage / nothing ran.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/ci_guard_steps.sh"
command -v jq > /dev/null 2>&1 || { echo "ci_guards: jq missing" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2

case "${1:-}" in
    --help|-h) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --list) shift; exec bash "$LIB" list "$@" ;;
    --sha) shift; exec bash "$LIB" sha "$@" ;;
    --check-coverage) shift; exec bash "$LIB" check-coverage "$@" ;;
esac
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --only|--step-timeout)
            [ "$#" -ge 2 ] || { echo "ci_guards: $1 needs a value" >&2; exit 2; }
            args+=("$1" "$2"); shift 2 ;;
        --stream|--no-stream) args+=("$1"); shift ;;
        -*) echo "ci_guards: unknown flag $1" >&2; exit 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
exec bash "$LIB" run "${args[@]}"
