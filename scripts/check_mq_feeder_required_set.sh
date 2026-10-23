#!/usr/bin/env bash
# check_mq_feeder_required_set.sh -- the check the merge-queue feeder requires on a PR head must be a job a PR head runs (PMAT-5843).
# The feeder required x86-main, which runs only in merge_group/push, so it refused every CLEAN PR as "x86-main is skipped".
# The case table is mq_feeder.sh --self-test: required set ∩ merge_group-only jobs of ci.yml = ∅, plus a planted overlap that must be caught.
#   bash scripts/check_mq_feeder_required_set.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(bash "$ROOT/scripts/mq_feeder.sh" --self-test 2>&1)"
rc=$?
if [ "$rc" -eq 0 ] && grep -q '^ok: self-test' <<< "$out" && grep -q "static: " "$ROOT/scripts/mq_feeder.sh"; then
    echo "check_mq_feeder_required_set: PASS (feeder required set disjoint from merge-queue-only jobs)"
    exit 0
fi
printf '%s\n' "$out" | tail -n 8
echo "check_mq_feeder_required_set: FAIL rc=$rc"
exit 1
