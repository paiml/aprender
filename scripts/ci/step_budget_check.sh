#!/usr/bin/env bash
# step_budget_check.sh -- flag any CI step that ran longer than 1.5x its budget (ci/step-budgets.tsv).
# Operator 2026-10-19: "per-step budgets, with CI flagging any step >1.5x its budget". Reads the run's jobs
# API (GraphQL-free REST, one call). Prints one line per breach; exit 0 always on a measured run, 2 when the
# run cannot be read (not_measured, never a pass). A LAB signal: run on demand or by the nightly LAB lane, never in a PR or merge job (C324 rule 5/6).
#   bash scripts/ci/step_budget_check.sh <run_id>      bash scripts/ci/step_budget_check.sh --self-test
set -uo pipefail
BUD=${STEP_BUDGETS:-ci/step-budgets.tsv}

# breaches: stdin = "job<TAB>step<TAB>seconds" rows; $BUD rows; prints "BREACH job | step | Ns > 1.5 x Bs"
breaches() {
  awk -F'\t' 'NR==FNR { if ($0 !~ /^#/ && NF >= 3) b[$1 FS $2] = $3; next }
    ($1 FS $2) in b { lim = b[$1 FS $2] * 1.5; if ($3 > lim) printf "BREACH %s | %s | %ds > 1.5 x %ds\n", $1, $2, $3, b[$1 FS $2] }' "$BUD" -
}

if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  printf 'j\ts\t100\n' > "$t/b.tsv"
  r1=$(printf 'j\ts\t151\n' | BUD="$t/b.tsv" breaches)
  r2=$(printf 'j\ts\t150\n' | BUD="$t/b.tsv" breaches)
  r3=$(printf 'other\ts\t999\n' | BUD="$t/b.tsv" breaches)
  [ -n "$r1" ] && [ -z "$r2" ] && [ -z "$r3" ] || { echo "FAIL: 151>1.5x100 must breach; 150 and an unbudgeted step must not"; exit 1; }
  echo "ok: step_budget_check case table"; exit 0
fi

run=${1:?usage: step_budget_check.sh <run_id>|--self-test}
rows=$(gh api "repos/${GITHUB_REPOSITORY:-paiml/aprender}/actions/runs/$run/jobs?per_page=100" --jq \
  '.jobs[] | .name as $j | .steps[]? | select(.started_at and .completed_at) | [$j, .name, ((.completed_at|fromdate) - (.started_at|fromdate))] | @tsv') \
  || { echo "NOT_MEASURED: jobs API read failed for run $run"; exit 2; }
out=$(printf '%s\n' "$rows" | breaches)
if [ -n "$out" ]; then printf '%s\n' "$out" | while read -r l; do echo "::error::$l"; done; else echo "no step over 1.5x budget (run $run)"; fi
