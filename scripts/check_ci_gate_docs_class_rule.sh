#!/usr/bin/env bash
# check_ci_gate_docs_class_rule.sh -- `gate` and `ci / gate` accept an absent
# section only on a class=docs run, and only for the sections a docs run skips
# (#4472).
#
# On class=docs (scripts/ci_change_class.sh, the `change-class` job) x86-main runs
# only the doc-content sections, and determinism does not run. The two verdict
# jobs then report sov.gate and determinism-compare as "not-triggered: docs-only".
# The rule this guard holds them to:
#   * docs counts ONLY when change-class SUCCEEDED and said class=docs; an empty
#     class, a failed job or class=full judges every section as before;
#   * guard-tree and guard-cargo must still succeed on a docs run, and so must
#     workspace-test (gate) and x86-main itself (ci / gate);
#   * nothing else is ever excused.
#
# It does not re-implement the rule. It EXTRACTS the two steps' `run:` scripts
# from .github/workflows/ci.yml and executes them for every row, so it judges the
# code the verdict jobs run. A step it cannot find is ENV rc=2, never a pass.
#
#   check_ci_gate_docs_class_rule.sh              the case table over ci.yml
#   check_ci_gate_docs_class_rule.sh --self-test  planted wrong rules must go RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
WF="${GATE_RULE_WORKFLOW:-$ROOT/.github/workflows/ci.yml}"
GATE_STEP="Check required sections"
CIGATE_STEP="The sovereign-ci gate section's result"

# step_run <workflow> <step name> -> that step's `run: |` body, dedented; empty if absent
step_run() {
    awk -v want="- name: $2" '
        index($0, want) && !found { found = 1; ind = -1; next }
        found && !body && /^ *run: \|/ { body = 1; next }
        found && !body && /^ *- (name|uses):/ { exit }
        body {
            if ($0 ~ /^[[:space:]]*$/) { print ""; next }
            match($0, /^ */)
            if (ind < 0) ind = RLENGTH
            if (RLENGTH < ind) exit
            print substr($0, ind + 1)
        }' "$1"
}

DOCS_X86='{"guard-tree":{"result":"success"},"guard-cargo":{"result":"success"},"vendored-schemas":{"result":"success"}}'
GT_FAIL='{"guard-tree":{"result":"failure"},"guard-cargo":{"result":"success"}}'
GC_MISS='{"guard-tree":{"result":"success"}}'
FULL_X86='{"guard-tree":{"result":"success"},"guard-cargo":{"result":"success"},"sov.gate":{"result":"success"}}'
FULL_DET='{"determinism-compare":{"result":"success"}}'

# run_gate <script> CLASS_JOB CLASS X86 DET WT -> exit status
run_gate() {
    CLASS_JOB=$2 CLASS=$3 CLASS_REASON=t X86=$4 DET=$5 WT=$6 YM='' EVT=pull_request \
        SCOPE=success IN_SCOPE=false TABLE=skipped bash -c "$1" > /dev/null 2>&1
}
# run_cigate <script> CLASS_JOB CLASS X86_RESULT RESULTS -> exit status
run_cigate() {
    CLASS_JOB=$2 CLASS=$3 CLASS_REASON=t X86_RESULT=$4 RESULTS=$5 bash -c "$1" > /dev/null 2>&1
}

table() { # table <workflow> -> 0 iff every row holds, 2 when a step is missing
    local g c bad=0 n=0 want got label
    g=$(step_run "$1" "$GATE_STEP")
    c=$(step_run "$1" "$CIGATE_STEP")
    [ -n "$g" ] && [ -n "$c" ] || { printf 'ENV   no "%s" or "%s" step in %s -- cannot judge, not a pass\n' "$GATE_STEP" "$CIGATE_STEP" "$1" >&2; return 2; }
    row() { # row WANT LABEL got-status
        n=$((n + 1)); want=$1 label=$2 got=$3
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %-4s %s\n' "$n" "$want" "$label"
        else printf 'FAIL  row %-2s wanted %s, got %s: %s\n' "$n" "$want" "$got" "$label" >&2; bad=1; fi
    }
    st() { if "$@"; then echo pass; else echo fail; fi; }
    row pass "gate: docs, guard-tree+guard-cargo ok, no sov.gate/determinism" "$(st run_gate "$g" success docs "$DOCS_X86" '' success)"
    row fail "gate: docs, guard-tree failed" "$(st run_gate "$g" success docs "$GT_FAIL" '' success)"
    row fail "gate: docs, guard-cargo missing" "$(st run_gate "$g" success docs "$GC_MISS" '' success)"
    row fail "gate: docs, workspace-test failed" "$(st run_gate "$g" success docs "$DOCS_X86" '' failure)"
    row fail "gate: docs output from a FAILED change-class" "$(st run_gate "$g" failure docs "$DOCS_X86" '' success)"
    row fail "gate: empty class (no decision), docs-shaped results" "$(st run_gate "$g" success '' "$DOCS_X86" '' success)"
    row fail "gate: class=full, docs-shaped results" "$(st run_gate "$g" success full "$DOCS_X86" '' success)"
    row fail "gate: class=full, determinism-compare missing" "$(st run_gate "$g" success full "$FULL_X86" '' success)"
    row pass "gate: class=full, every section ok" "$(st run_gate "$g" success full "$FULL_X86" "$FULL_DET" success)"
    row pass "gate: change-class skipped/failed, every section ok" "$(st run_gate "$g" failure '' "$FULL_X86" "$FULL_DET" success)"
    row pass "ci/gate: docs, x86-main + guard-tree + guard-cargo ok" "$(st run_cigate "$c" success docs success "$DOCS_X86")"
    row fail "ci/gate: docs, x86-main failed" "$(st run_cigate "$c" success docs failure "$DOCS_X86")"
    row fail "ci/gate: docs, x86-main ok but guard-tree failed" "$(st run_cigate "$c" success docs success "$GT_FAIL")"
    row fail "ci/gate: docs, x86-main ok but guard-cargo missing" "$(st run_cigate "$c" success docs success "$GC_MISS")"
    row fail "ci/gate: docs, results empty" "$(st run_cigate "$c" success docs success '')"
    row fail "ci/gate: docs output from a FAILED change-class" "$(st run_cigate "$c" failure docs success "$DOCS_X86")"
    row fail "ci/gate: empty class, no sov.gate" "$(st run_cigate "$c" success '' success "$DOCS_X86")"
    row fail "ci/gate: class=full, no sov.gate" "$(st run_cigate "$c" success full success "$DOCS_X86")"
    row pass "ci/gate: class=full, sov.gate ok" "$(st run_cigate "$c" success full success "$FULL_X86")"
    return "$bad"
}

case "${1:-}" in -h | --help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== gate docs-class rule: planted wrong rules must turn the table RED ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/gate-docs-class.XXXXXX") || exit 2
    trap 'rm -rf -- "${d:?}"' EXIT
    [ -f "$WF" ] || { printf 'ENV   %s is missing\n' "$WF" >&2; exit 2; }
    # trusts the output alone: a docs class from a change-class job that did not succeed
    sed 's/case "\$CLASS_JOB:\$CLASS:\$sec" in/case "success:$CLASS:$sec" in/; s/if \[ "\$CLASS_JOB" = success \] && \[ "\$CLASS" = docs \]/if [ "$CLASS" = docs ]/' "$WF" > "$d/nojob.yml"
    # the permissive mutant: on docs every section is excused, guard-tree included
    sed 's/success:docs:sov.gate | success:docs:determinism-compare)/success:docs:*)/' "$WF" > "$d/loose.yml"
    # ci / gate trusts x86-main's job result alone, not the doc-content sections
    awk '/for sec in guard-tree guard-cargo; do/ { skip = 1 } skip && /^ *done$/ { skip = 0; next } !skip' "$WF" > "$d/cig.yml"
    # ci / gate drops guard-cargo from the doc-content sections it still requires
    sed 's/for sec in guard-tree guard-cargo; do/for sec in guard-tree; do/' "$WF" > "$d/cigc.yml"
    # gate ends the whole check at the first excused section instead of skipping it
    awk '/success:docs:sov.gate \| success:docs:determinism-compare\)/ { hit = 1 } hit && /continue ;;/ { sub(/continue ;;/, "exit 0 ;;"); hit = 0 } 1' "$WF" > "$d/exit.yml"
    printf 'jobs: {}\n' > "$d/none.yml"
    bad=0
    for m in nojob loose cig cigc exit; do
        if cmp -s "$WF" "$d/$m.yml"; then printf 'FAIL  the %s mutant did not apply (its anchor is gone)\n' "$m"; bad=1; continue; fi
        if table "$d/$m.yml" > "$d/out" 2>&1; then printf 'FAIL  the planted %s rule passed the table\n' "$m"; bad=1
        else printf 'ok    the planted %-5s rule is RED: %s row(s), e.g. %s\n' "$m" "$(grep -c '^FAIL' "$d/out")" "$(grep -m1 '^FAIL' "$d/out" | cut -c7-90)"; fi
    done
    rc=0; table "$d/none.yml" > /dev/null 2>&1 || rc=$?
    if [ "$rc" = 2 ]; then printf 'ok    a workflow with no verdict steps is ENV rc=2, never a pass\n'
    else printf 'FAIL  no verdict steps gave rc=%s\n' "$rc"; bad=1; fi
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== gate / ci / gate excuse sov.gate+determinism-compare only on a succeeded class=docs (check_ci_gate_docs_class_rule.sh) ==="
[ -f "$WF" ] || { printf 'ENV   %s is missing\n' "$WF" >&2; exit 2; }
table "$WF"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
