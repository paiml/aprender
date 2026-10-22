#!/usr/bin/env bash
# advisory_step_selftest.sh -- the CI step "Advisories must pass" must go RED when the gate (or any
# self-test before it) exits non-zero (PMAT-5792, ADG-008). Extracts the step's run block from
# .github/workflows/ci.yml, runs it in a throwaway repo whose origin/main holds stub scripts, and
# requires the step's exit status to equal the stub's. A mutant (gate line removed) must NOT be red.
#
#   advisory_step_selftest.sh [WORKFLOW]   WORKFLOW defaults to .github/workflows/ci.yml AND ci/sections.yml
set -euo pipefail

wfs=("${1:-.github/workflows/ci.yml}")
[ -n "${1:-}" ] || wfs+=(ci/sections.yml)   # the generated-from copy is held to the same rows
d=$(mktemp -d)
# shellcheck disable=SC2064
trap "rm -rf -- '${d:?}'" EXIT

extract() { # WORKFLOW -> the run block, de-indented
    awk '/^      - name: Advisories must pass/ {on=1; next}
         on && /^      - name:/ {exit}
         on && /^        run: \|/ {run=1; next}
         on && run {sub(/^          /, ""); print}' "$1"
}

# stubs: each script exits with the status in <name>.rc (default 0)
make_repo() { # DIR -> repo with scripts/ci stubs committed on refs/remotes/origin/main
    local r=$1 f
    mkdir -p "$r/scripts/ci"
    for f in advisory_diff advisory_gate advisory_gate_selftest advisory_step_selftest advisory_positive_control check_sov_advisories_off; do
        printf '#!/usr/bin/env bash\nexit "$(cat "$(dirname "$0")/%s.rc" 2>/dev/null || echo 0)"\n' "$f" >"$r/scripts/ci/$f.sh"
    done
    git -C "$r" init -q
    git -C "$r" add -A
    git -C "$r" -c user.name=t -c user.email=t@t write-tree >/dev/null
    local tree c
    tree=$(git -C "$r" write-tree)
    c=$(git -C "$r" -c user.name=t -c user.email=t@t commit-tree "$tree" -m stub)
    git -C "$r" update-ref refs/remotes/origin/main "$c"
}

run_step() { # SCRIPTFILE REPO -> step exit status
    local rc=0
    (cd "$2" && env -u RUNNER_TEMP TMPDIR="$d" GITHUB_TOKEN=x MG_BASE_SHA="${MG:-}" bash -c "$(cat "$1")" >/dev/null 2>&1) || rc=$?
    echo "$rc"
}

# $1: gate rc, $2: first self-test rc -- set in the BASE copy (what the step actually runs) and in the tree
# $6: "" | right | wrong -- MG_BASE_SHA the merge queue names: right = origin/main's sha, wrong = another sha
case_() { # name want gate_rc selftest_rc script [mg]
    local name=$1 want=$2 got r
    r=$(mktemp -d "$d/repo-XXXXXX")
    make_repo "$r"
    printf '%s' "$3" >"$r/scripts/ci/advisory_gate.rc"
    printf '%s' "$4" >"$r/scripts/ci/advisory_diff.rc"
    git -C "$r" add -A
    local tree c
    tree=$(git -C "$r" write-tree)
    c=$(git -C "$r" -c user.name=t -c user.email=t@t commit-tree "$tree" -m stub2)
    git -C "$r" update-ref refs/remotes/origin/main "$c"
    case "${6:-}" in right) MG=$c ;; wrong) MG=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb ;; *) MG= ;; esac
    got=$(MG=$MG run_step "$5" "$r")
    if [ "$got" -ne "$want" ]; then
        echo "SELF-TEST FAIL: ${wf}: $name wanted step rc=$want, got $got" >&2
        bad=1
    fi
}

bad=0
for wf in "${wfs[@]}"; do
    extract "$wf" >"$d/step.sh"
    [ -s "$d/step.sh" ] || { echo "SELF-TEST FAIL: could not extract the 'Advisories must pass' step from $wf" >&2; exit 1; }
    case_ 'gate rc 75 -> step RED with 75' 75 75 0 "$d/step.sh"
    case_ 'gate rc 1 -> step RED with 1' 1 1 0 "$d/step.sh"
    case_ 'advisory_diff self-test rc 3 -> step RED with 3' 3 0 3 "$d/step.sh"
    case_ 'all green -> step green' 0 0 0 "$d/step.sh"
    # the BASE gate runs before any PR-tree self-test: with both red, the gate's status wins
    case_ 'gate runs before the PR-tree self-tests (gate 75, self-test 3 -> 75)' 75 75 3 "$d/step.sh"
    # merge_group: origin/main must be the ref the queue named; the PR tree cannot re-point it
    case_ 'merge_group with origin/main == base_sha -> green' 0 0 0 "$d/step.sh" right
    case_ 'merge_group with origin/main != base_sha -> RED 75' 75 0 0 "$d/step.sh" wrong
    # mutant: the gate invocation removed from the step; a failing gate must then (wrongly) pass -- proving the cases above bind to it
    grep -v 'advisory_gate.sh"$' "$d/step.sh" >"$d/mut.sh"
    cmp -s "$d/step.sh" "$d/mut.sh" && { echo "SELF-TEST FAIL: ${wf}: mutant did not change the step (gate line not found)" >&2; bad=1; }
    case_ 'MUTANT (gate line removed) must NOT be red on a failing gate' 0 75 0 "$d/mut.sh"
    # mutant 2: the gate moved to the END of the step (the shape the opus lane flagged): with both red, the self-test's 3 now wins
    { grep -v 'advisory_gate.sh"$' "$d/step.sh"; grep 'advisory_gate.sh"$' "$d/step.sh"; } >"$d/mut2.sh"
    case_ 'MUTANT (gate last) must lose to the self-test status' 3 75 3 "$d/mut2.sh"
    # mutant 3: the merge_group base_sha guard removed (the whole if-block): a re-pointed origin/main must then (wrongly) pass
    sed '/^if \[ -n "\${MG_BASE_SHA/,/^fi$/d' "$d/step.sh" >"$d/mut3.sh"
    cmp -s "$d/step.sh" "$d/mut3.sh" && { echo "SELF-TEST FAIL: ${wf}: mutant 3 did not change the step (MG guard not found)" >&2; bad=1; }
    case_ 'MUTANT (MG guard removed) must NOT be red on a wrong base_sha' 0 0 0 "$d/mut3.sh" wrong
done
if [ "$bad" -eq 0 ]; then
    echo "advisory_step self-test: ${#wfs[@]} workflow(s) x (7 cases + 3 mutants) ok"
fi
exit "$bad"
