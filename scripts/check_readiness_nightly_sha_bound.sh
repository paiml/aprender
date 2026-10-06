#!/usr/bin/env bash
# check_readiness_nightly_sha_bound.sh -- a readiness-nightly run is credited only to the
# commit it measured (#4719).
#
# WHY. The judge (scripts/release/nightly_train.sh, lane `readiness`) credits a
# workflow_run run of readiness-nightly.yml to the run's head commit. GitHub stamps a
# workflow_run run with main HEAD at trigger time (GITHUB_SHA), but the job measures the
# upstream run's commit (workflow_run.head_sha). models-nightly runs for hours, so main
# moves under it: without a bound, a green readiness run for X is credited to Y, which
# nobody measured. The inputs job therefore runs a workflow_run only when the two agree;
# otherwise it is skipped, which the judge reads as not_measured.
#
# THE RULE. The `if:` of job `inputs` is exactly CANON below, whitespace collapsed. Exact
# match, not a substring: `head_sha == github.sha` moved outside the workflow_run arm (an
# `||` alternative) still contains the words and binds nothing. Changing the trigger gate
# means changing CANON here in the same PR, in front of review.
#
#   bash scripts/check_readiness_nightly_sha_bound.sh               # gate the repo's workflow
#   bash scripts/check_readiness_nightly_sha_bound.sh --file F      # gate a fixture
#   bash scripts/check_readiness_nightly_sha_bound.sh --self-test   # case table + mutants
#
# EXIT 0 bound · 1 not bound · 2 cannot judge (no file, no inputs job, no `if:`).
# bash + awk only: the clean-room fleet carries no yq.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF="$HERE/../.github/workflows/readiness-nightly.yml"
CANON="github.event_name == 'workflow_dispatch' || (github.event.workflow_run.head_branch == 'main' && github.event.workflow_run.head_repository.full_name == github.repository && github.event.workflow_run.event == 'schedule' && github.event.workflow_run.head_sha == github.sha)"

# the `if:` of job `inputs`, folded scalar or one line, whitespace collapsed; empty if absent
inputs_if() {
    awk '
        /^jobs:[[:space:]]*$/ { injobs = 1; next }
        injobs && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { job = $1; sub(/:$/, "", job); inif = 0; next }
        injobs && job == "inputs" && /^    if:/ {
            v = $0; sub(/^    if:[[:space:]]*/, "", v)
            if (v ~ /^>-?[[:space:]]*$/) { inif = 1; next }
            print v; exit
        }
        inif && /^      / { s = $0; sub(/^[[:space:]]+/, "", s); printf "%s ", s; next }
        inif { exit }
    ' "$1" | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//'
}

# gate FILE -> prints one line, returns 0/1/2
gate() {
    local f="$1" got
    [ -f "$f" ] || { echo "NOT_MEASURED: no workflow at $f"; return 2; }
    got="$(inputs_if "$f")"
    [ -n "$got" ] || { echo "NOT_MEASURED: no if: on job inputs in $f"; return 2; }
    if [ "$got" = "$CANON" ]; then echo "PASS: readiness inputs if: binds workflow_run.head_sha to github.sha"; return 0; fi
    printf 'FAIL: readiness inputs if: is not the sha-bound gate\n  want: %s\n  got:  %s\n' "$CANON" "$got"
    return 1
}

self_test() {
    local pass=0 fail=0 rc o
    tmp="$(mktemp -d)" || exit 2
    trap 'rm -rf -- "${tmp:?}"' EXIT
    fx() {   # fx NAME IF-BODY-LINES -> a minimal workflow with that folded if: on job inputs
        printf 'on:\n  workflow_run:\n    workflows: [models-nightly]\njobs:\n  inputs:\n    runs-on: x\n    if: >-\n%s\n    steps:\n      - run: true\n  readiness:\n    needs: inputs\n    if: needs.inputs.outputs.ok == '"'"'yes'"'"'\n' "$2" > "$tmp/$1.yml"
    }
    row() {   # row NAME WANT_RC FILE
        o="$(gate "$3")"; rc=$?
        if [ "$rc" = "$2" ]; then printf '  ok    %-40s rc=%s\n' "$1" "$rc"; pass=$((pass + 1))
        else printf '  BROKE %-40s want rc=%s got %s\n%s\n' "$1" "$2" "$rc" "$o"; fail=$((fail + 1)); fi
    }
    local A="      github.event_name == 'workflow_dispatch' ||"
    local B="      (github.event.workflow_run.head_branch == 'main' &&"
    local C="       github.event.workflow_run.head_repository.full_name == github.repository &&"
    local D="       github.event.workflow_run.event == 'schedule' &&"
    local E="       github.event.workflow_run.head_sha == github.sha)"
    fx bound "$A"$'\n'"$B"$'\n'"$C"$'\n'"$D"$'\n'"$E"
    row bound_passes 0 "$tmp/bound.yml"
    # must-RED: the review finding itself -- the bound dropped (#4809 as it was)
    fx unbound "$A"$'\n'"$B"$'\n'"$C"$'\n'"       github.event.workflow_run.event == 'schedule')"
    row bound_dropped_is_red 1 "$tmp/unbound.yml"
    # must-RED: the words present, but as an alternative that binds nothing
    fx ored "$A"$'\n'"$B"$'\n'"$C"$'\n'"       github.event.workflow_run.event == 'schedule') ||"$'\n'"      github.event.workflow_run.head_sha == github.sha"
    row bound_as_alternative_is_red 1 "$tmp/ored.yml"
    # must-RED: the schedule gate (#4809) dropped while the bound stays
    fx nosched "$A"$'\n'"$B"$'\n'"$C"$'\n'"$E"
    row schedule_gate_dropped_is_red 1 "$tmp/nosched.yml"
    # one-line if: is read the same as the folded one
    printf 'jobs:\n  inputs:\n    if: %s\n    runs-on: x\n' "$CANON" > "$tmp/oneline.yml"
    row one_line_if_passes 0 "$tmp/oneline.yml"
    # the bound on ANOTHER job does not count
    printf 'jobs:\n  inputs:\n    runs-on: x\n  readiness:\n    if: %s\n' "$CANON" > "$tmp/otherjob.yml"
    row bound_on_other_job_unmeasured 2 "$tmp/otherjob.yml"
    row missing_file_unmeasured 2 "$tmp/nope.yml"
    # the real workflow, and a mutant of it with the bound removed
    row repo_workflow_bound 0 "$WF"
    sed -e "s/^\(       github\.event\.workflow_run\.event == 'schedule'\) \&\&$/\1)/" \
        -e '/^       github\.event\.workflow_run\.head_sha == github\.sha)$/d' "$WF" > "$tmp/m.yml"
    [ "$(diff "$WF" "$tmp/m.yml" | grep -c '^[<>]')" -eq 3 ] || : > "$tmp/m.yml"   # exactly the two lines, or no mutant
    if cmp -s "$WF" "$tmp/m.yml"; then printf '  BROKE %-40s mutant not built (vacuous)\n' repo_mutant_bound_removed; fail=$((fail + 1))
    else row repo_mutant_bound_removed_is_red 1 "$tmp/m.yml"; fi
    echo "self-test: $pass passed, $fail broke"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --file) [ -n "${2:-}" ] || { echo "usage: $0 [--file F | --self-test]" >&2; exit 2; }; gate "$2"; exit $? ;;
    "") gate "$WF"; exit $? ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "usage: $0 [--file F | --self-test]" >&2; exit 2 ;;
esac
