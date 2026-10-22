#!/usr/bin/env bash
# check_ci_failfast.sh -- a RED merge_group job must cancel its run (operator 2026-10-19, refs #5394).
# x86-main and workspace-test-shard each carry a `gh run cancel` step guarded by
# `failure() && github.event_name == 'merge_group'`, and the shard job holds `actions: write`.
# Also runs scripts/ci/step_budget_check.sh --self-test and checks every budget row has 3 fields.
#   bash scripts/check_ci_failfast.sh [workflow] | --self-test
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

# job_has_failfast <workflow> <job>: 0 when the job block holds the guarded cancel step
job_has_failfast() {
  awk -v job="$2" '
    $0 ~ "^  " job ":" { on = 1; next }
    on && /^  [a-z0-9-]+:/ { on = 0 }
    on && /if: .*failure\(\).*github\.event_name == .merge_group./ { g = 1 }
    on && g && /gh run cancel "\$GITHUB_RUN_ID"/ { ok = 1 }
    END { exit ok ? 0 : 1 }' "$1"
}

check() {
  local wf=$1 rc=0 j
  for j in x86-main workspace-test-shard; do
    job_has_failfast "$wf" "$j" || { echo "FAIL: job $j has no merge_group failure() -> gh run cancel step"; rc=1; }
  done
  awk '/^  workspace-test-shard:/{on=1;next} on&&/^  [a-z0-9-]+:/{on=0} on&&/actions: write/{f=1} END{exit f?0:1}' "$wf" \
    || { echo "FAIL: workspace-test-shard lacks 'actions: write' (gh run cancel would 403)"; rc=1; }
  return $rc
}

case "${1:-}" in -h|--help) echo "usage: check_ci_failfast.sh [workflow] | --self-test"; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
  check .github/workflows/ci.yml || { echo "FAIL: the real workflow is not fail-fast"; exit 1; }
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  grep -v 'gh run cancel' .github/workflows/ci.yml > "$t/mut.yml"
  check "$t/mut.yml" > /dev/null && { echo "FAIL: a workflow with the cancel step removed still passed"; exit 1; }
  bash scripts/ci/step_budget_check.sh --self-test || exit 1
  awk -F'\t' '!/^#/ && NF != 3 { bad = 1 } END { exit bad }' ci/step-budgets.tsv || { echo "FAIL: ci/step-budgets.tsv row without 3 fields"; exit 1; }
  echo "ok: check_ci_failfast case table"; exit 0
fi
check "${1:-.github/workflows/ci.yml}" && echo "ok: merge_group failures cancel the run"
