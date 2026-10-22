#!/usr/bin/env bash
# guard_lane_stop.sh -- the C324 item 5 stop for the LAB check scripts/check_guard_lane.sh (#5920): a LAB check red or
# unmeasured for 7 days stops running until its owner brings it back.
#
# guards-nightly carries many guards in ONE job, so the whole workflow cannot be disabled the way
# scripts/mutants_weekly_stop.sh disables mutants-nightly. This stop is per STEP: it reads the conclusion this step had
# in the finished runs of the last 8 days and decides before tonight's run whether the step runs.
#
# stop    = at least 7 finished runs in the window carry the step and NONE of them concluded success. A step that was
#           skipped by this stop is not green, so a stopped check stays stopped.
# run     = anything else: fewer than 7 records (a new check cannot stop in its first week), a green night in the
#           window, a manual dispatch (the owner's way back: dispatch the workflow; a green dispatch ends the stop),
#           or a listing that failed -- a read we could not make is not_measured and decides nothing.
#
#   guard_lane_stop.sh --run         in the workflow: GH_TOKEN, GITHUB_REPOSITORY, GITHUB_EVENT_NAME, GITHUB_OUTPUT
#   guard_lane_stop.sh --self-test   the decision case table; a failed listing must be run, never stop
set -uo pipefail
WINDOW_DAYS=8
MIN_RECORDS=7
STEP_NAME="ci_guards cargo-free lane self-test (#5920, LAB)"
JOB_NAME="guards-nightly"

# decide <records> <green> -> prints run|stop|error; rc 0 run/stop, 1 error
decide() {
    case "$1" in ''|*[!0-9]*) echo error; return 1 ;; esac
    case "$2" in ''|*[!0-9]*) echo error; return 1 ;; esac
    [ "$2" -gt "$1" ] && { echo error; return 1; }
    [ "$1" -ge "$MIN_RECORDS" ] && [ "$2" -eq 0 ] && { echo stop; return 0; }
    echo run
}

if [ "${1:-}" = "--self-test" ]; then
    bad=0
    # records|green|want
    while IFS='|' read -r r g want; do
        got=$(decide "$r" "$g")
        if [ "$got" = "$want" ]; then printf 'ok    %s record(s), %s green -> %s\n' "${r:-<none>}" "${g:-<none>}" "$want"
        else printf 'FAIL  %s record(s), %s green: wanted %s, got %s\n' "$r" "$g" "$want" "$got" >&2; bad=1; fi
    done <<'CASES'
0|0|run
6|0|run
7|0|stop
8|0|stop
7|1|run
8|8|run
7|7|run
x|0|error
7||error
7|abc|error
3|5|error
CASES
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

[ "${1:-}" = "--run" ] || { printf 'usage: %s --run | --self-test\n' "$(basename "$0")" >&2; exit 2; }
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_OUTPUT:?}"
emit() { printf 'decision=%s\n' "$1" >> "$GITHUB_OUTPUT"; echo "guard-lane stop: decision $1 ($2)"; }
if [ "${GITHUB_EVENT_NAME:-}" != schedule ]; then emit run "event ${GITHUB_EVENT_NAME:-unset}: a dispatch always runs the check"; exit 0; fi
API="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions"
SINCE=$(date -u -d "-${WINDOW_DAYS} days" +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002
auth=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
ids=$(curl -sSf "${auth[@]}" "$API/workflows/guards-nightly.yml/runs?branch=main&status=completed&created=%3E%3D${SINCE}&per_page=30" \
    | jq -r '.workflow_runs[].id') || { emit run "run listing failed: not_measured, no stop decision"; exit 0; }
n=0 green=0
for id in $ids; do
    c=$(curl -sSf "${auth[@]}" "$API/runs/$id/jobs?per_page=100" \
        | jq -r --arg j "$JOB_NAME" --arg s "$STEP_NAME" '.jobs[] | select(.name == $j) | .steps[] | select(.name == $s) | .conclusion') \
        || { emit run "job listing for run $id failed: not_measured, no stop decision"; exit 0; }
    [ -n "$c" ] || continue   # the run predates the step: no record, not a red night
    n=$((n + 1))
    [ "$c" = success ] && green=$((green + 1))
done
if ! D=$(decide "$n" "$green"); then emit run "bad counts $n/$green: not_measured"; exit 0; fi
emit "$D" "$n finished run(s) carry the step since $SINCE, $green green"
