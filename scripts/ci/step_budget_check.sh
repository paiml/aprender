#!/usr/bin/env bash
# step_budget_check.sh -- flag any CI step that ran longer than 1.5x its budget (ci/step-budgets.tsv).
# Operator 2026-10-19: "per-step budgets, with CI flagging any step >1.5x its budget". Reads the run's jobs
# API (GraphQL-free REST, one call). Prints one line per breach. Exit 0: measured, no breach. Exit 1: a breach. Exit 2: the
# run, the budgets or any step row could not be read (not_measured, never a pass). A LAB signal: run on demand or by the nightly LAB lane, never in a PR or merge job (C324 rule 5/6).
#   bash scripts/ci/step_budget_check.sh <run_id>      bash scripts/ci/step_budget_check.sh --self-test
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
BUD=${STEP_BUDGETS:-ci/step-budgets.tsv}

# check_rows: stdin = "job<TAB>step<TAB>seconds" rows, $BUD = budgets. Prints "SEEN job | step" for each budgeted step
# that appears and "BREACH job | step | Ns > 1.5 x Bs" for each one over 1.5x. Exit 2 (not measured) when $BUD is missing,
# empty or has no 3-field row, or when no budgeted step appears in the rows: nothing measured is never a pass.
check_rows() {
  [ -s "$BUD" ] && awk -F'\t' '!/^#/ && NF >= 3 { ok = 1 } END { exit ok ? 0 : 1 }' "$BUD" || { echo "NOT_MEASURED: budget file $BUD missing, empty or without a 3-field row"; return 2; }
  local seen
  seen=$(awk -F'\t' 'NR==FNR { if ($0 !~ /^#/ && NF >= 3) b[$1 FS $2] = $3; next }
    ($1 FS $2) in b { printf "SEEN %s | %s\n", $1, $2; lim = b[$1 FS $2] * 1.5; if ($3 > lim) printf "BREACH %s | %s | %ds > 1.5 x %ds\n", $1, $2, $3, b[$1 FS $2] }' "$BUD" -) || return 2
  printf '%s\n' "$seen" | grep -q '^SEEN ' || { echo "NOT_MEASURED: no budgeted step appears in the run's rows"; return 2; }
  printf '%s\n' "$seen" | grep '^BREACH ' && return 1
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  printf 'j\ts\t100\n' > "$t/b.tsv"
  rc() { printf '%b' "$2" | BUD="$1" check_rows > /dev/null; echo $?; }
  [ "$(rc "$t/b.tsv" 'j\ts\t151\n')" = 1 ] || { echo "FAIL: 151 > 1.5x100 must breach (exit 1)"; exit 1; }
  [ "$(rc "$t/b.tsv" 'j\ts\t150\n')" = 0 ] || { echo "FAIL: 150 is not a breach (exit 0)"; exit 1; }
  [ "$(rc "$t/b.tsv" 'other\ts\t999\n')" = 2 ] || { echo "FAIL: only an unbudgeted step in the rows is not measured (exit 2)"; exit 1; }
  [ "$(rc "$t/b.tsv" '')" = 2 ] || { echo "FAIL: no rows is not measured (exit 2)"; exit 1; }
  : > "$t/empty.tsv"; printf '# only a comment\n' > "$t/comment.tsv"
  [ "$(rc "$t/empty.tsv" 'j\ts\t999\n')" = 2 ] || { echo "FAIL: an empty budget file is not measured (exit 2)"; exit 1; }
  [ "$(rc "$t/comment.tsv" 'j\ts\t999\n')" = 2 ] || { echo "FAIL: a comment-only budget file is not measured (exit 2)"; exit 1; }
  [ "$(rc "$t/nofile.tsv" 'j\ts\t999\n')" = 2 ] || { echo "FAIL: a missing budget file is not measured (exit 2)"; exit 1; }
  echo "ok: step_budget_check case table"; exit 0
fi

run=${1:?usage: step_budget_check.sh <run_id>|--self-test}
rows=$(gh api "repos/${GITHUB_REPOSITORY:-paiml/aprender}/actions/runs/$run/jobs?per_page=100" --jq \
  '.jobs[] | .name as $j | .steps[]? | select(.started_at and .completed_at) | [$j, .name, ((.completed_at|fromdate) - (.started_at|fromdate))] | @tsv') \
  || { echo "NOT_MEASURED: jobs API read failed for run $run"; exit 2; }
out=$(printf '%s\n' "$rows" | check_rows); rc=$?
case $rc in
  0) echo "no step over 1.5x budget (run $run)" ;;
  1) printf '%s\n' "$out" | while read -r l; do echo "::error::$l"; done ;;
  *) printf '%s\n' "$out" ;;
esac
exit $rc
