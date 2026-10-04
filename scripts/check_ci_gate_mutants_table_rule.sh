#!/usr/bin/env bash
# check_ci_gate_mutants_table_rule.sh -- the `gate` job's survivor-table rule (#4587 A', operator 2026-09-29).
#
# For a head ref in ci/mutants-table-refs.txt the `mutants` section hands over to the mutants-shard matrix, so
# `gate` must then require mutants-table == success, and on every pull_request the scope decision itself must
# have run. This guard EXTRACTS the GATE-MUTANTS-TABLE-RULE block from .github/workflows/ci.yml and executes it
# for every (event, scope result, in_scope, table result) row, so it judges the code the gate runs. A missing
# block is ENV rc=2, never a pass. It also runs the survivor-table checker's own case table.
#
#   check_ci_gate_mutants_table_rule.sh              the case table over ci.yml's block + the checker's table
#   check_ci_gate_mutants_table_rule.sh --self-test  planted weak rules must turn the table RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
WF="${GATE_RULE_WORKFLOW:-$ROOT/.github/workflows/ci.yml}"

extract() { awk '/GATE-MUTANTS-TABLE-RULE-BEGIN/{f=1; next} /GATE-MUTANTS-TABLE-RULE-END/{f=0} f' "$1" | sed -E 's/^ {10}//'; }

table() { # table <workflow> -> 0 iff every row holds
    local blk bad=0 n=0 evt scope ins tab want got
    blk=$(extract "$1")
    [ -n "$blk" ] || { printf 'ENV   no GATE-MUTANTS-TABLE-RULE block in %s -- cannot judge, not a pass\n' "$1" >&2; return 2; }
    while read -r evt scope ins tab want; do
        [ -n "$evt" ] || continue
        [ "$ins" = - ] && ins=""
        n=$((n + 1)); got=0
        EVT=$evt SCOPE=$scope IN_SCOPE=$ins TABLE=$tab bash -c "$blk" >/dev/null 2>&1 || got=1
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %-12s scope=%-9s in=%-5s table=%-9s -> %s\n' "$n" "$evt" "$scope" "${ins:--}" "$tab" "$([ "$want" = 0 ] && echo pass || echo FAIL)"
        else printf 'FAIL  row %-2s %-12s scope=%-9s in=%-5s table=%-9s wanted %s, got %s\n' "$n" "$evt" "$scope" "${ins:--}" "$tab" "$want" "$got" >&2; bad=1; fi
    done <<'ROWS'
pull_request success   true  success   0
pull_request success   true  failure   1
pull_request success   true  skipped   1
pull_request success   true  cancelled 1
pull_request success   false skipped   0
pull_request success   false success   1
pull_request success   -     skipped   1
pull_request failure   -     skipped   1
pull_request skipped   -     skipped   1
pull_request cancelled -     skipped   1
merge_group  skipped   -     skipped   0
push         skipped   -     skipped   0
merge_group  success   true  success   1
merge_group  skipped   -     failure   1
ROWS
    return "$bad"
}

case "${1:-}" in -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== gate survivor-table rule: planted weak rules must turn the table RED ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/gate-table-rule.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "${1:?}" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    bad=0
    plant() { # plant <name> <rule body>: a weak rule that must NOT pass the table
        printf '          # --- GATE-MUTANTS-TABLE-RULE-BEGIN ---\n%s\n          # --- GATE-MUTANTS-TABLE-RULE-END ---\n' "$2" > "$d/$1.yml"
        if table "$d/$1.yml" > "$d/out" 2>&1; then printf 'FAIL  the planted rule "%s" passed the table\n' "$1"; bad=1
        else printf 'ok    planted "%s" is RED: %s row(s)\n' "$1" "$(grep -c '^FAIL' "$d/out")"; fi
    }
    plant table-failure-only '          [ "$TABLE" != failure ]'
    plant ignore-scope '          case "$TABLE" in success|skipped) : ;; *) exit 1 ;; esac'
    plant trust-in-scope-only '          [ "$IN_SCOPE" != true ] || [ "$TABLE" = success ]'
    printf 'jobs: {}\n' > "$d/none.yml"
    rc=0; table "$d/none.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && printf 'ok    a workflow with no rule block is ENV rc=2, never a pass\n' || { printf 'FAIL  no rule block gave rc=%s\n' "$rc"; bad=1; }
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== gate requires the survivor table for a listed head ref (check_ci_gate_mutants_table_rule.sh) ==="
[ -f "$WF" ] || { printf 'ENV   %s is missing\n' "$WF" >&2; exit 2; }
table "$WF"; rc=$?
echo "=== the survivor-table checker's case table (scripts/ci/mutants_survivor_table.py --self-test) ==="
python3 "$ROOT/scripts/ci/mutants_survivor_table.py" --self-test || rc=1
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
