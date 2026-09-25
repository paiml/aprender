#!/usr/bin/env bash
# check_ci_gate_docs_only_rule.sh -- the `gate` job accepts a SKIPPED guard-cargo or
# determinism-compare only when the `changes` job SUCCEEDED and said docs_only=true
# (#3668).
#
# guard-cargo and determinism are skipped on a docs-only PR (every touched path
# under docs/roadmaps/ or docs/audits/, scripts/ci_docs_only.sh). Their `if:` is
# fail-closed, so a skipped/failed `changes` job makes them RUN; the gate must be
# fail-closed too: any skip without a successful docs_only=true is a failure.
#
# This guard does not re-implement the rule. It EXTRACTS the block between the
# GATE-DOCS-ONLY-RULE-BEGIN/END markers from .github/workflows/ci.yml and executes
# it for every (docs_only, changes, guard-cargo, determinism-compare) row, so it
# judges the code the gate runs. A missing or empty block is ENV rc=2, never a pass.
#
#   check_ci_gate_docs_only_rule.sh              the case table over ci.yml's block
#   check_ci_gate_docs_only_rule.sh --self-test  planted wrong rules must go RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
WF="${GATE_RULE_WORKFLOW:-$ROOT/.github/workflows/ci.yml}"

# extract <workflow> -> the rule block, dedented; empty when the markers are absent
extract() {
    awk '/GATE-DOCS-ONLY-RULE-BEGIN/{f=1; next} /GATE-DOCS-ONLY-RULE-END/{f=0} f' "$1" | sed -E 's/^ {10}//'
}

# verdict <block> <docs> <changes> <gc> <dc> -> exit status of the block; "-" is empty
verdict() {
    local docs=$2
    [ "$docs" = "-" ] && docs=""
    DOCS=$docs CHG=$3 GC=$4 DC=$5 bash -c "$1" >/dev/null 2>&1
}

table() { # table <workflow> -> 0 iff every row holds
    local blk bad=0 n=0 docs chg gc dc want got
    blk=$(extract "$1")
    [ -n "$blk" ] || { printf 'ENV   no GATE-DOCS-ONLY-RULE block in %s -- cannot judge, not a pass\n' "$1" >&2; return 2; }
    while read -r docs chg gc dc want; do
        [ -n "$docs" ] || continue
        n=$((n + 1)); got=0; verdict "$blk" "$docs" "$chg" "$gc" "$dc" || got=1
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s docs=%-5s changes=%-8s gc=%-9s dc=%-9s -> %s\n' "$n" "$docs" "$chg" "$gc" "$dc" "$([ "$want" = 0 ] && echo pass || echo FAIL)"
        else printf 'FAIL  row %-2s docs=%-5s changes=%-8s gc=%-9s dc=%-9s wanted %s, got %s\n' "$n" "$docs" "$chg" "$gc" "$dc" "$want" "$got" >&2; bad=1; fi
    done <<'ROWS'
true  success success   success   0
true  success skipped   skipped   0
false success skipped   success   1
false success success   skipped   1
-     skipped skipped   skipped   1
true  failure skipped   skipped   1
true  success failure   skipped   1
true  success cancelled skipped   1
true  success skipped   failure   1
false success success   success   0
-     skipped success   success   0
-     failure success   success   0
false success failure   success   1
-     skipped success   failure   1
ROWS
    return "$bad"
}

case "${1:-}" in -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== gate docs-only rule: planted wrong rules must turn the table RED ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/gate-docs-rule.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    # the pre-#3668 rule: require success, never accept a skip
    printf '          # --- GATE-DOCS-ONLY-RULE-BEGIN ---\n          [ "$GC" = success ] && [ "$DC" = success ] || exit 1\n          # --- GATE-DOCS-ONLY-RULE-END ---\n' > "$d/old.yml"
    # the permissive mutant: a skip always passes, whatever `changes` said
    printf '          # --- GATE-DOCS-ONLY-RULE-BEGIN ---\n          for r in "$GC" "$DC"; do case "$r" in success|skipped) ;; *) exit 1 ;; esac; done\n          # --- GATE-DOCS-ONLY-RULE-END ---\n' > "$d/loose.yml"
    # trusts the output alone: a docs_only=true from a job that did not succeed
    printf '          # --- GATE-DOCS-ONLY-RULE-BEGIN ---\n          for r in "$GC" "$DC"; do case "$r:$DOCS" in success:*|skipped:true) ;; *) exit 1 ;; esac; done\n          # --- GATE-DOCS-ONLY-RULE-END ---\n' > "$d/nochg.yml"
    printf 'jobs: {}\n' > "$d/none.yml"
    bad=0
    for m in old loose nochg; do
        if table "$d/$m.yml" > "$d/out" 2>&1; then printf 'FAIL  the planted %s rule passed the table\n' "$m"; bad=1
        else printf 'ok    the planted %-5s rule is RED: %s row(s), e.g. %s\n' "$m" "$(grep -c '^FAIL' "$d/out")" "$(grep -m1 '^FAIL' "$d/out" | cut -c7-80)"; fi
    done
    rc=0; table "$d/none.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && printf 'ok    a workflow with no rule block is ENV rc=2, never a pass\n' || { printf 'FAIL  no rule block gave rc=%s\n' "$rc"; bad=1; }
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== gate accepts a skipped guard-cargo/determinism-compare only on a docs-only PR (check_ci_gate_docs_only_rule.sh) ==="
[ -f "$WF" ] || { printf 'ENV   %s is missing\n' "$WF" >&2; exit 2; }
table "$WF"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
