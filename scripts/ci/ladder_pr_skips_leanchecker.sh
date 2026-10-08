#!/usr/bin/env bash
# ladder_pr_skips_leanchecker.sh — planted rows for operator ruling C312 (#4578): a PR run of
# the provable-ladder skips the leanchecker re-check, and so does a push to main (amended 2026-10-06,
# item 5); the full run is the nightly schedule (provable-ladder-nightly.yml).
#
# Drives the real scripts/ci/provable_ladder_discharge.sh with a stub pv that records its argv
# and a stub build.sh, per event, and checks which pv command ran. Then the summary-fresh rows
# on a throwaway git repo. Then plants two mutants — every event mapped to `pr` (the nightly
# never runs the leanchecker) and push mapped back to `full` (the leanchecker returns to the
# push path) — and requires a row to catch each, so the table is proven able to see both.
#
# EXIT 0 every row holds and the mutant is caught · 1 otherwise
#
#   bash scripts/ci/ladder_pr_skips_leanchecker.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SUT="$ROOT/scripts/ci/provable_ladder_discharge.sh"
TD="$(mktemp -d)"
trap 'rm -rf "${TD:?}"' EXIT

# A Lean dir whose build.sh exits $BUILD_RC, and a pv that appends its argv to $TD/argv.
mkdir -p "$TD/lean"
printf '#!/usr/bin/env bash\nexit "${BUILD_RC:-0}"\n' > "$TD/lean/build.sh"
printf '#!/usr/bin/env bash\necho "$*" >> "%s/argv"\n' "$TD" > "$TD/pv"
chmod +x "$TD/pv"
# A flock that records it was taken, then runs the command: the .lake must be locked on every path.
mkdir -p "$TD/bin"
printf "#!/usr/bin/env bash\ntouch \"%s/locked\"\nshift\nexec \"\$@\"\n" "$TD" > "$TD/bin/flock"
chmod +x "$TD/bin/flock"

fail=0
row() { # row <name> <got> <want>
    if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: got '$2', want '$3'"; fail=1; fi
}

# pv_argv <sut> <event> [build rc] -> "<rc>|<pv argv, or none>|<locked or unlocked>"
pv_argv() {
    local rc=0
    rm -f "$TD/argv" "$TD/locked"
    ( cd "$TD" && PATH="$TD/bin:$PATH" EVENT="$2" BUILD_RC="${3:-0}" PV="$TD/pv" LEAN=lean LOCK="$TD/lock" \
        LOG="$TD/log" bash "$1" discharge > /dev/null 2>&1 ) || rc=$?
    echo "$rc|$(cat "$TD/argv" 2>/dev/null || echo none)|$([ -e "$TD/locked" ] && echo locked || echo unlocked)"
}

table() { # table <sut> <label> — the event rows
    local s="$1" l="$2"
    row "$l pull_request: check --strict --comparator, no --leanchecker" \
        "$(pv_argv "$s" pull_request)" "0|discharge check lean --strict --comparator|locked"
    row "$l merge_group: same as a PR" \
        "$(pv_argv "$s" merge_group)" "0|discharge check lean --strict --comparator|locked"
    row "$l push: same as a PR (the push path skips the re-check)" \
        "$(pv_argv "$s" push)" "0|discharge check lean --strict --comparator|locked"
    row "$l schedule (nightly): discharge run" "$(pv_argv "$s" schedule)" "0|discharge run lean|locked"
    row "$l workflow_dispatch: discharge run" "$(pv_argv "$s" workflow_dispatch)" "0|discharge run lean|locked"
    row "$l empty event fails safe to the full run" "$(pv_argv "$s" '')" "0|discharge run lean|locked"
    row "$l PR, build.sh declines (rc 2): rc 2, pv never runs" \
        "$(pv_argv "$s" pull_request 2)" "2|none|locked"
    row "$l PR, build.sh fails (rc 1): rc 1, pv never runs" \
        "$(pv_argv "$s" pull_request 1)" "1|none|locked"
}

table "$SUT" real

# The wiring: the table above sets EVENT itself, so check the section hands the real event to
# both steps that call the script. An unset EVENT would be safe (full) but would silently put
# the 75-minute leanchecker back on every PR.
sec="$(awk '/^  provable-ladder:/{on=1; next} on && /^  [a-z]/{exit} on' "$ROOT/ci/sections.yml")"
row "wiring: both steps get EVENT from github.event_name" \
    "$(grep -c '^ *EVENT: \${{ github.event_name }}$' <<< "$sec")" 2
row "wiring: both steps call provable_ladder_discharge.sh (discharge, summary-fresh)" \
    "$(grep -cE 'scripts/ci/provable_ladder_discharge\.sh (discharge|summary-fresh)$' <<< "$sec")" 2

# summary-fresh rows, on a repo shaped like ours.
G="$TD/repo"
mkdir -p "$G/lean"
echo 'theorem t : True := trivial' > "$G/lean/A.lean"
git -C "$G" init -q
git -C "$G" add lean
git -C "$G" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm lean
sha="$(git -C "$G" rev-parse HEAD:lean)"
fresh() { # fresh <event> -> rc of summary-fresh
    local rc=0
    ( cd "$G" && EVENT="$1" LEAN=lean SUMMARY=summary.json bash "$SUT" summary-fresh > /dev/null 2>&1 ) || rc=$?
    echo "$rc"
}
commit_summary() {
    printf '{\n  "tree_sha": "%s",\n  "leanchecker_exit": 0\n}\n' "$1" > "$G/summary.json"
    git -C "$G" add summary.json
    git -C "$G" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm summary
}
commit_summary "$sha"
row "PR summary-fresh: tree_sha = HEAD:lean -> 0" "$(fresh pull_request)" 0
commit_summary "0000000000000000000000000000000000000000"
row "PR summary-fresh: stale tree_sha -> 1" "$(fresh pull_request)" 1
git -C "$G" rm -q summary.json
git -C "$G" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm drop
row "PR summary-fresh: no committed summary -> 1" "$(fresh pull_request)" 1
commit_summary "$sha"
row "push summary-fresh: same as a PR, tree_sha = HEAD:lean -> 0" "$(fresh push)" 0
row "schedule summary-fresh: unchanged after run -> 0" "$(fresh schedule)" 0
echo '{}' > "$G/summary.json"
row "schedule summary-fresh: run rewrote it -> 1" "$(fresh schedule)" 1
git -C "$G" checkout -q -- summary.json

# The nightly is the only place the leanchecker now runs, so check it is wired: a schedule
# trigger, the provable-ladder section, and a verdict read from results.json (fat_driver skips a
# continue-on-error section, so the job's own status would be green on a RED ladder).
NW="$ROOT/.github/workflows/provable-ladder-nightly.yml"
row "nightly: provable-ladder-nightly.yml has a schedule trigger" \
    "$(grep -cE "^    - cron: '[0-9]+ [0-9]+ \* \* \*'" "$NW" 2> /dev/null || echo 0)" 1
sections_arg="--sections 'provable-ladder'"
row "nightly: it runs the provable-ladder section through fat_driver" \
    "$(grep -cF -- "$sections_arg" "$NW" 2> /dev/null || echo 0)" 1
row "nightly: its verdict reads the section's result, not the job status" \
    "$(grep -cF '[env.SEC].result' "$NW" 2> /dev/null || echo 0)" 1

# The verdict step itself, RUN on stub results files: a grep of its text cannot see a step that
# also accepts `failure`, or one that passes when fat_driver never wrote a result.
# step_lines: the verdict step's lines, from its name to the next step.
step_lines() { awk -v n="$VNAME" 'index($0, n) { on = 1 } on && /^      - / && !index($0, n) { exit } on' "$NW"; }
VNAME="name: The ladder's verdict is the section's result, not the job status"
vblock() { step_lines | sed -n '/^        run: |$/,$p' | sed '1d; s/^          //'; }
vsec() { step_lines | sed -n 's/^          SEC: //p'; }
# Stub results.json files, one per case. `absent` is no file at all.
mkdir -p "$TD/res"
while IFS='=' read -r name body; do
    printf '%s\n' "$body" > "$TD/res/$name.json"
done << 'FIXTURES'
success={"provable-ladder": {"result": "success"}}
failure={"provable-ladder": {"result": "failure"}}
skipped={"provable-ladder": {"result": "skipped"}}
cancelled={"provable-ladder": {"result": "cancelled"}}
other={"x": {"result": "success"}}
garbage=not json
FIXTURES
vstep() { # vstep <fixture name, or absent> -> rc of the verdict block
    local d="$TD/v" rc=0
    rm -rf "${d:?}"
    mkdir -p "$d/fat"
    [ "$1" = absent ] || cp -- "$TD/res/${1:?}.json" "$d/fat/results.json"
    vblock > "$d/verdict.sh"
    RUNNER_TEMP="$d" GITHUB_STEP_SUMMARY="$d/summary" SEC="$(vsec)" bash "$d/verdict.sh" > /dev/null 2>&1 || rc=$?
    echo "$rc"
}
if ! command -v jq > /dev/null; then
    echo "FAIL  nightly verdict rows: no jq on PATH, so the verdict step cannot be run (not measured)"
    fail=1
elif [ -z "$(vblock)" ]; then
    echo "FAIL  nightly verdict rows: the verdict step's run block was not found in $NW"
    fail=1
else
    row "nightly verdict: it judges the provable-ladder section" "$(vsec)" provable-ladder
    row "nightly verdict: success -> 0" "$(vstep success)" 0
    row "nightly verdict: failure -> 1" "$(vstep failure)" 1
    row "nightly verdict: skipped -> 1" "$(vstep skipped)" 1
    row "nightly verdict: cancelled -> 1" "$(vstep cancelled)" 1
    row "nightly verdict: another section green, the ladder absent -> 1" "$(vstep other)" 1
    row "nightly verdict: fat_driver wrote no results.json -> 1" "$(vstep absent)" 1
    row "nightly verdict: unreadable results.json -> 1" "$(vstep garbage)" 1
fi
# The step must run whenever the job was not cancelled, and read the file the run step writes.
row "nightly: the verdict step runs unless cancelled" "$(step_lines | grep -c '^        if: \${{ !cancelled() }}$')" 1
results_file="$(printf '%s' '"$RUNNER_TEMP/fat/results.json"')"
n_writes="$(grep -cF -- "--results $results_file" "$NW" || true)"
n_refs="$(grep -cF -- "$results_file" "$NW" || true)"
row "nightly: fat_driver writes the results file the verdict reads" "$n_writes|$n_refs" "1|2"
row "nightly: the run step calls fat_driver run" "$(grep -cx '          python3 scripts/ci/fat_driver.py run' "$NW" || true)" 1

# mutant <name> <sed expr> <event> <the row's passing output>: the planted defect must change the
# script, and the named event's row must no longer pass on it.
mutant() {
    local m
    sed "$2" "$SUT" > "$TD/mutant.sh"
    if cmp -s "$SUT" "$TD/mutant.sh"; then
        echo "FAIL  mutant ($1): the sed did not change the script (pattern drifted)"; fail=1; return
    fi
    m="$(pv_argv "$TD/mutant.sh" "$3")"
    if [ "$m" = "$4" ]; then
        echo "FAIL  mutant ($1) survived the $3 row"; fail=1
    else
        echo "ok    mutant ($1) caught by the $3 row ($m)"
    fi
}
# Every event takes the PR path, so the leanchecker never runs anywhere, the nightly included.
all_pr='s/^        pull_request | merge_group | push) echo pr ;;$/        *) echo pr ;;/'
mutant "all events -> pr" "$all_pr" schedule "0|discharge run lean|locked"
# The amendment undone: push falls back to the full run and the leanchecker is on the push path.
push_full='s/^        pull_request | merge_group | push) echo pr ;;$/        pull_request | merge_group) echo pr ;;/'
mutant "push -> full" "$push_full" push "0|discharge check lean --strict --comparator|locked"

exit "$fail"
