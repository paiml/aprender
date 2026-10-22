#!/usr/bin/env bash
# guard_lane_stop.sh -- the C324 item 5 stop for the LAB check scripts/check_guard_lane.sh (#5920): a LAB check red or
# unmeasured for 7 days stops running until its owner brings it back.
#
# guards-nightly carries many guards in ONE job, so the whole workflow cannot be disabled the way
# scripts/mutants_weekly_stop.sh disables mutants-nightly. This stop is per STEP: before tonight's run it asks whether
# the step concluded success in ANY finished main run of the last 7 days.
#
# stop    = the check has been live for 7 days or more (ACTIVE_SINCE) and the last 7 days hold NO green night. The
#           window is a span of time, not a count of records, so a dropped or late cron night cannot hold the stop off.
#           Unmeasured counts as red: a finished run without the step, a skipped step (a stopped check stays stopped),
#           a run whose jobs cannot be read and a run listing that fails all add no green. Reading the history is the
#           stop's own measurement, so an unreadable history stops the check for the night; the next night that can
#           read a green one runs it again, and nothing is lost but one night.
# run     = a green night in the window; fewer than 7 days live (a new check cannot stop in its first week); a manual
#           dispatch (the owner's way back: dispatch the workflow, and a green dispatch ends the stop).
#
#   guard_lane_stop.sh --run         in the workflow: GH_TOKEN, GITHUB_REPOSITORY, GITHUB_EVENT_NAME, GITHUB_OUTPUT
#   guard_lane_stop.sh --self-test   the decision case table
set -uo pipefail
WINDOW_DAYS=7
# The first night the check ran in guards-nightly (UTC date). Bump it if the landing slips: a date before the landing
# would stop a check that has never run.
ACTIVE_SINCE="2026-10-24"
STEP_NAME="ci_guards cargo-free lane self-test (#5920, LAB)"
JOB_NAME="guards-nightly"

# decide <whole days the check has been live> <green nights in the window> -> prints run|stop
# A green count that is not a number is an unreadable history: not green, so it can stop.
decide() {
    case "$1" in ''|*[!0-9]*) echo run; return 0 ;; esac   # an unreadable clock decides nothing
    case "$2" in ''|*[!0-9]*) set -- "$1" 0 ;; esac
    [ "$1" -ge "$WINDOW_DAYS" ] && [ "$2" -eq 0 ] && { echo stop; return 0; }
    echo run
}

if [ "${1:-}" = "--self-test" ]; then
    bad=0
    # days live|green nights in the window|want
    while IFS='|' read -r d g want; do
        got=$(decide "$d" "$g")
        if [ "$got" = "$want" ]; then printf 'ok    live %s day(s), green %s -> %s\n' "${d:-<none>}" "${g:-<none>}" "$want"
        else printf 'FAIL  live %s day(s), green %s: wanted %s, got %s\n' "$d" "$g" "$want" "$got" >&2; bad=1; fi
    done <<'CASES'
0|0|run
6|0|run
7|0|stop
30|0|stop
7|1|run
30|3|run
30||stop
30|abc|stop
30|-1|stop
x|0|run
CASES
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

[ "${1:-}" = "--run" ] || { printf 'usage: %s --run | --self-test\n' "$(basename "$0")" >&2; exit 2; }
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_OUTPUT:?}"
emit() { printf 'decision=%s\n' "$1" >> "$GITHUB_OUTPUT"; echo "guard-lane stop: decision $1 ($2)"; }
if [ "${GITHUB_EVENT_NAME:-}" != schedule ]; then emit run "event ${GITHUB_EVENT_NAME:-unset}: a dispatch always runs the check"; exit 0; fi
NOW=${GUARD_LANE_STOP_NOW:-$(date -u +%s)}  # bashrs disable-line=DET002
live=$(( (NOW - $(date -u -d "${ACTIVE_SINCE}T00:00:00Z" +%s)) / 86400 )) || live=
SINCE=$(date -u -d "@$((NOW - WINDOW_DAYS * 86400))" +%Y-%m-%dT%H:%M:%SZ)
API="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions"
auth=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
green=0 why=""
if ids=$(curl -sSf "${auth[@]}" "$API/workflows/guards-nightly.yml/runs?branch=main&status=completed&created=%3E%3D${SINCE}&per_page=30" \
        | jq -r '.workflow_runs[].id'); then
    for id in $ids; do
        # a run whose jobs cannot be read, or that lacks the step, adds no green
        c=$(curl -sSf "${auth[@]}" "$API/runs/$id/jobs?per_page=100" \
            | jq -r --arg j "$JOB_NAME" --arg s "$STEP_NAME" '.jobs[] | select(.name == $j) | .steps[] | select(.name == $s) | .conclusion') \
            || { why="$why run $id unreadable;"; continue; }
        [ "$c" = success ] && green=$((green + 1))
    done
else
    why=" run listing unreadable (counted as no green);"
fi
D=$(decide "$live" "$green")
emit "$D" "live ${live:-?} day(s), $green green night(s) since $SINCE;$why"
