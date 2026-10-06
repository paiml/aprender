#!/usr/bin/env bash
# ladder_pr_skips_leanchecker.sh — planted rows for operator ruling C312 (#4578): a PR run of
# the provable-ladder skips the leanchecker re-check; a push to main runs it in full.
#
# Drives the real scripts/ci/provable_ladder_discharge.sh with a stub pv that records its argv
# and a stub build.sh, per event, and checks which pv command ran. Then the summary-fresh rows
# on a throwaway git repo. Then plants the mutant (every event mapped to `pr`) and requires
# the full-run rows to catch it — so the table is proven able to see the defect.
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
    row "$l push: discharge run (with the leanchecker's replay)" "$(pv_argv "$s" push)" "0|discharge run lean|locked"
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
row "push summary-fresh: unchanged after run -> 0" "$(fresh push)" 0
echo '{}' > "$G/summary.json"
row "push summary-fresh: run rewrote it -> 1" "$(fresh push)" 1
git -C "$G" checkout -q -- summary.json

# The mutant: every event takes the PR path, so leanchecker never runs anywhere.
sed 's/^        pull_request | merge_group) echo pr ;;$/        *) echo pr ;;/' "$SUT" > "$TD/mutant.sh"
if cmp -s "$SUT" "$TD/mutant.sh"; then
    echo "FAIL  mutant: the sed did not change the script (pattern drifted)"; fail=1
else
    m="$(pv_argv "$TD/mutant.sh" push)"
    if [ "$m" = "0|discharge run lean|locked" ]; then
        echo "FAIL  mutant (all events -> pr) survived the push row"; fail=1
    else
        echo "ok    mutant (all events -> pr) caught by the push row ($m)"
    fi
fi

exit "$fail"
