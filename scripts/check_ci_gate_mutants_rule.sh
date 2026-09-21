#!/usr/bin/env bash
# check_ci_gate_mutants_rule.sh -- the `gate` job accepts a SKIPPED mutants job only
# when the skip is that job's own filter (#3676).
#
# mutants runs only on pull_request (`if: github.event_name == 'pull_request'`),
# so on merge_group, push and workflow_dispatch it is skipped by design. On a
# pull_request it must RUN: a diff with no .rs still succeeds, because its
# mutation steps skip and the job passes. The old rule read "success/skipped both
# pass", which also passed a pull_request whose mutants job never ran at all.
#
# This guard does not re-implement the rule. It EXTRACTS the block between the
# GATE-MUTANTS-RULE-BEGIN/END markers from .github/workflows/ci.yml and executes
# it for every (event, result) pair, so it judges the code the gate runs. A
# missing or empty block is ENV rc=2, never a pass.
#
#   check_ci_gate_mutants_rule.sh              the case table over ci.yml's block
#   check_ci_gate_mutants_rule.sh --self-test  the old rule, planted: it must go RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
WF="${GATE_RULE_WORKFLOW:-$ROOT/.github/workflows/ci.yml}"

# extract <workflow> -> the rule block, dedented; empty when the markers are absent
extract() {
    awk '/GATE-MUTANTS-RULE-BEGIN/{f=1; next} /GATE-MUTANTS-RULE-END/{f=0} f' "$1" | sed -E 's/^ {10}//'
}

# verdict <block> <event> <result> -> exit status of the block with EVT/MUT set
verdict() {
    EVT=$2 MUT=$3 bash -c "$1" >/dev/null 2>&1
}

table() { # table <workflow> -> 0 iff every row holds
    local blk bad=0 n=0 evt mut want got
    blk=$(extract "$1")
    [ -n "$blk" ] || { printf 'ENV   no GATE-MUTANTS-RULE block in %s -- cannot judge, not a pass\n' "$1" >&2; return 2; }
    while read -r evt mut want; do
        [ -n "$evt" ] || continue
        n=$((n + 1)); got=0; verdict "$blk" "$evt" "$mut" || got=1
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %-14s %-10s -> %s\n' "$n" "$evt" "$mut" "$([ "$want" = 0 ] && echo pass || echo FAIL)"
        else printf 'FAIL  row %-2s %-14s %-10s wanted %s, got %s\n' "$n" "$evt" "$mut" "$want" "$got" >&2; bad=1; fi
    done <<'ROWS'
pull_request      success    0
pull_request      skipped    1
pull_request      cancelled  1
pull_request      failure    1
merge_group       skipped    0
push              skipped    0
workflow_dispatch skipped    0
merge_group       success    0
merge_group       failure    1
merge_group       cancelled  1
ROWS
    return "$bad"
}

case "${1:-}" in -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== gate mutants rule: the old rule must turn the table RED ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/gate-rule.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    printf '          # --- GATE-MUTANTS-RULE-BEGIN ---\n          if [ "$MUT" = "failure" ]; then exit 1; fi\n          # --- GATE-MUTANTS-RULE-END ---\n' > "$d/old.yml"
    printf 'jobs: {}\n' > "$d/none.yml"
    bad=0
    if table "$d/old.yml" > "$d/out" 2>&1; then printf 'FAIL  the OLD rule ("success/skipped both pass") passed the table\n'; bad=1
    else printf 'ok    the OLD rule is RED: %s\n' "$(grep -c '^FAIL' "$d/out") row(s), e.g. $(grep -m1 '^FAIL' "$d/out" | cut -c7-70)"; fi
    rc=0; table "$d/none.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && printf 'ok    a workflow with no rule block is ENV rc=2, never a pass\n' || { printf 'FAIL  no rule block gave rc=%s\n' "$rc"; bad=1; }
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== gate accepts a skipped mutants job only when the skip is its filter (check_ci_gate_mutants_rule.sh) ==="
# the workflow is needed only here: --self-test builds its own fixtures (quorum lane 1 on #3688)
[ -f "$WF" ] || { printf 'ENV   %s is missing\n' "$WF" >&2; exit 2; }
table "$WF"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
