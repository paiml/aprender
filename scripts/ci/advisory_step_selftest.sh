#!/usr/bin/env bash
# advisory_step_selftest.sh -- the CI step "Advisories must pass" must go RED when the gate (or any
# self-test before it) exits non-zero (PMAT-5792, ADG-008). Extracts the step's run block from
# .github/workflows/ci.yml, runs it in a throwaway repo whose origin/main holds stub scripts, and
# requires the step's exit status to equal the stub's. A mutant (gate line removed) must NOT be red.
#
#   advisory_step_selftest.sh [WORKFLOW]   WORKFLOW defaults to .github/workflows/ci.yml
set -euo pipefail

wf=${1:-.github/workflows/ci.yml}
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
    (cd "$2" && env -u RUNNER_TEMP TMPDIR="$d" GITHUB_TOKEN=x bash -c "$(cat "$1")" >/dev/null 2>&1) || rc=$?
    echo "$rc"
}

# $1: gate rc, $2: first self-test rc -- set in the BASE copy (what the step actually runs) and in the tree
case_() { # name want gate_rc selftest_rc script
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
    got=$(run_step "$5" "$r")
    if [ "$got" -ne "$want" ]; then
        echo "SELF-TEST FAIL: $name wanted step rc=$want, got $got" >&2
        bad=1
    fi
}

bad=0
extract "$wf" >"$d/step.sh"
[ -s "$d/step.sh" ] || { echo "SELF-TEST FAIL: could not extract the 'Advisories must pass' step from $wf" >&2; exit 1; }
case_ 'gate rc 75 -> step RED with 75' 75 75 0 "$d/step.sh"
case_ 'gate rc 1 -> step RED with 1' 1 1 0 "$d/step.sh"
case_ 'advisory_diff self-test rc 3 -> step RED with 3' 3 0 3 "$d/step.sh"
case_ 'all green -> step green' 0 0 0 "$d/step.sh"
# mutant: the gate invocation removed from the step; a failing gate must then (wrongly) pass -- proving the cases above bind to it
grep -v 'advisory_gate.sh"$' "$d/step.sh" >"$d/mut.sh"
cmp -s "$d/step.sh" "$d/mut.sh" && { echo "SELF-TEST FAIL: mutant did not change the step (gate line not found)" >&2; bad=1; }
case_ 'MUTANT (gate line removed) must NOT be red on a failing gate' 0 75 0 "$d/mut.sh"
# mutant 2: the old `|| rc=$?` shape with the gate dropped from the chain is caught the same way (covered by the case above)
if [ "$bad" -eq 0 ]; then
    echo "advisory_step self-test: 4 cases + 1 mutant ok"
fi
exit "$bad"
