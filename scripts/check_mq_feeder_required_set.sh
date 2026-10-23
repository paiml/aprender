#!/usr/bin/env bash
# check_mq_feeder_required_set.sh -- the check the merge-queue feeder requires on a PR head must be a job a PR head runs (#6175).
# The feeder required x86-main, which runs only in merge_group/push, so it refused every CLEAN PR as "x86-main is skipped".
# Bare run: --self-test, then `mq_feeder.sh --check-required` against this tree's ci.yml. --self-test plants the incident
# (required x86-main, merge-queue-only, multi-line `if:`), an absent job, a missing file, one good workflow, and drives the
# feeder end to end to prove it refuses to act (exit 2 + inbox line) when the required check cannot exist on a PR head.
#   bash scripts/check_mq_feeder_required_set.sh [--self-test]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEEDER="${MQ_FEEDER_UNDER_TEST:-$ROOT/scripts/mq_feeder.sh}"
if [ "${1:-}" = --self-test ]; then
    d="$(mktemp -d)"; fail=0
    printf 'jobs:\n  pr-light:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/good.yml"
    printf 'jobs:\n  pr-light:\n    if: ${{ github.event_name != '"'pull_request'"' }}\n' > "$d/bad-mqonly.yml"
    printf 'jobs:\n  other:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/bad-absent.yml"
    printf 'jobs:\n  x86-main:\n    name: x86-main\n    if: >-\n      github.event_name != '"'pull_request'"'\n    runs-on: x\n  pr-light:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/incident.yml"
    bash "$FEEDER" --check-required "$d/good.yml" > /dev/null 2>&1 || { echo "FAIL good fixture refused"; fail=1; }
    bash "$FEEDER" --check-required "$d/bad-mqonly.yml" > /dev/null 2>&1 && { echo "FAIL mq-only required check accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/bad-absent.yml" > /dev/null 2>&1 && { echo "FAIL absent required check accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/missing.yml" > /dev/null 2>&1 && { echo "FAIL missing workflow accepted"; fail=1; }
    GUARD_CHECK_NAME=x86-main bash "$FEEDER" --check-required "$d/incident.yml" > /dev/null 2>&1 && { echo "FAIL incident (x86-main, multi-line if) accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/incident.yml" > /dev/null 2>&1 || { echo "FAIL incident workflow refused for pr-light"; fail=1; }
    # end to end: the deployed feeder judges the workflow it is handed, refuses to act, and says so in the inbox
    mkdir -p "$d/st"; : > "$d/inbox"
    GUARD_CHECK_NAME=x86-main MQ_FEEDER_CI_FILE="$d/incident.yml" MQ_FEEDER_STATE="$d/st" MQ_FEEDER_INBOX="$d/inbox" bash "$FEEDER" > /dev/null 2>&1; rc=$?
    [ "$rc" -eq 2 ] || { echo "FAIL feeder acted on an unsatisfiable required check (rc=$rc)"; fail=1; }
    grep -q 'runs only in merge_group/push' "$d/inbox" || { echo "FAIL feeder refused silently (no inbox line)"; fail=1; }
    rm -rf "${d:?}"
    [ "$fail" -eq 0 ] && echo "check_mq_feeder_required_set: self-test ok (2 good, 4 planted bad, 1 end-to-end)"
    exit "$fail"
fi
bash "$0" --self-test || exit 1
out="$(bash "$FEEDER" --check-required "$ROOT/.github/workflows/ci.yml" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ]; then echo "check_mq_feeder_required_set: PASS ($out)"; exit 0; fi
printf '%s\n' "$out" | tail -n 4
echo "check_mq_feeder_required_set: FAIL rc=$rc"
exit 1
