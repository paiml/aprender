#!/usr/bin/env bash
# scripts/lab/ci_failfast_audit.sh -- NO merge_group job may cancel its own run (operator 2026-10-27, refs #6438 throughput).
# The 2026-10-19 fail-fast (`gh run cancel` on the first red merge_group job) is removed: it hid the cause behind
# cancelled siblings and cost merges. This audit now fails if a `gh run cancel "$GITHUB_RUN_ID"` step comes back.
# Also runs scripts/ci/step_budget_check.sh --self-test and checks every budget row has 3 fields.
#   bash scripts/lab/ci_failfast_audit.sh [workflow] | --self-test
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2

# check <workflow>: 0 when no step cancels the run it is in
check() {
  if grep -n 'gh run cancel "\$GITHUB_RUN_ID"' "$1"; then
    echo "FAIL: $1 has a step that cancels its own run (fail-fast removed 2026-10-27)"; return 1
  fi
  return 0
}

case "${1:-}" in -h|--help) echo "usage: scripts/lab/ci_failfast_audit.sh [workflow] | --self-test"; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
  check .github/workflows/ci.yml || { echo "FAIL: the real workflow cancels its own run"; exit 1; }
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  { cat .github/workflows/ci.yml; printf '      - run: gh run cancel "$GITHUB_RUN_ID" --repo x\n'; } > "$t/mut.yml"
  check "$t/mut.yml" > /dev/null && { echo "FAIL: a workflow with a self-cancel step still passed"; exit 1; }
  bash scripts/ci/step_budget_check.sh --self-test || exit 1
  awk -F'\t' '!/^#/ && NF != 3 { bad = 1 } END { exit bad }' ci/step-budgets.tsv || { echo "FAIL: ci/step-budgets.tsv row without 3 fields"; exit 1; }
  echo "ok: ci_failfast_audit case table"; exit 0
fi
check "${1:-.github/workflows/ci.yml}" && echo "ok: no merge_group step cancels its own run"
