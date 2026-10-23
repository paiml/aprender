#!/usr/bin/env bash
# check_mq_feeder_required_set.sh -- the check the merge-queue feeder requires on a PR head must be a job a PR head runs (#6175).
# The feeder required x86-main, which runs only in merge_group/push, so it refused every CLEAN PR as "x86-main is skipped".
# Bare run: asks `mq_feeder.sh --check-required` about the REAL ci.yml, then runs --self-test (planted bad and good fixtures).
#   bash scripts/check_mq_feeder_required_set.sh [--self-test]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEEDER="$ROOT/scripts/mq_feeder.sh"
if [ "${1:-}" = --self-test ]; then
    d="$(mktemp -d)"; fail=0
    printf 'jobs:\n  pr-light:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/good.yml"
    printf 'jobs:\n  pr-light:\n    if: ${{ github.event_name != '"'pull_request'"' }}\n' > "$d/bad-mqonly.yml"
    printf 'jobs:\n  other:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/bad-absent.yml"
    bash "$FEEDER" --check-required "$d/good.yml" > /dev/null 2>&1 || { echo "FAIL good fixture refused"; fail=1; }
    bash "$FEEDER" --check-required "$d/bad-mqonly.yml" > /dev/null 2>&1 && { echo "FAIL mq-only required check accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/bad-absent.yml" > /dev/null 2>&1 && { echo "FAIL absent required check accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/missing.yml" > /dev/null 2>&1 && { echo "FAIL missing workflow accepted"; fail=1; }
    rm -rf "${d:?}"
    [ "$fail" -eq 0 ] && echo "check_mq_feeder_required_set: self-test ok (1 good, 3 planted bad)"
    exit "$fail"
fi
bash "$0" --self-test || exit 1
out="$(bash "$FEEDER" --check-required 2>&1)"; rc=$?
if [ "$rc" -eq 0 ]; then echo "check_mq_feeder_required_set: PASS ($out)"; exit 0; fi
printf '%s\n' "$out" | tail -n 4
echo "check_mq_feeder_required_set: FAIL rc=$rc"
exit 1
