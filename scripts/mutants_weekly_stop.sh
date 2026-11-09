#!/usr/bin/env bash
# mutants_weekly_stop.sh -- the C324 item 5 stop for the weekly mutants-nightly (#5228): a LAB check that is red or
# unmeasured for 7 days stops running until its owner brings it back (`gh workflow enable mutants-nightly.yml`).
#
# The window is 8 days, not 7. The scheduled run is created Sunday 03:00Z and job `stop` reads the listing up to 3 h
# later, so last Sunday's run was created MORE than 7 days before the read: a 7-day window misses it, and one red
# after a green week would disable the workflow (C314 on #5229). 8 days holds last week's run and not the one before,
# so the stop fires only when this run AND last week's (and any dispatch between) are all non-green.
#
#   mutants_weekly_stop.sh --run         in the workflow: RESULT, GH_TOKEN, GITHUB_REPOSITORY from the env
#   mutants_weekly_stop.sh --self-test   the decision case table; a failed listing must be an error, never a stop
set -uo pipefail
WINDOW_DAYS=8

# since_epoch <now, epoch s> -> the window's start (epoch s). --self-test checks it; since_iso formats it for --run.
since_epoch() { echo $(($1 - WINDOW_DAYS * 86400)); }
# since_iso <now, epoch s> -> the window's start as the listing's created>= value.
since_iso() { date -u -d "@$(since_epoch "$1")" +%Y-%m-%dT%H:%M:%SZ; }
# now_epoch -> the one wall-clock read; MUTANTS_STOP_NOW pins it (epoch s) for a reproducible run.
now_epoch() { echo "${MUTANTS_STOP_NOW:-$(date -u +%s)}"; }  # bashrs disable-line=DET002

# decide <result> <listing rc> <green runs in the window> -> prints keep|stop|error; rc 0 keep/stop, 1 error
decide() {
    [ "$1" = success ] && { echo keep; return 0; }
    [ "$2" = 0 ] || { echo error; return 1; }
    case "$3" in ''|*[!0-9]*) echo error; return 1 ;; esac
    [ "$3" -gt 0 ] && { echo keep; return 0; }
    echo stop
}

if [ "${1:-}" = "--self-test" ]; then
    bad=0
    # result|listing rc|green runs in the window|want
    while IFS='|' read -r r lrc n want; do
        got=$(decide "$r" "$lrc" "$n")
        if [ "$got" = "$want" ]; then printf 'ok    %s, listing rc %s, %s green -> %s\n' "$r" "$lrc" "${n:-<none>}" "$want"
        else printf 'FAIL  %s, listing rc %s, %s green: wanted %s, got %s\n' "$r" "$lrc" "${n:-<none>}" "$want" "$got" >&2; bad=1; fi
    done <<'CASES'
success|0|0|keep
success|22|x|keep
failure|0|1|keep
failure|0|2|keep
cancelled|0|1|keep
failure|0|0|stop
cancelled|0|0|stop
skipped|0|0|stop
failure|22|0|error
failure|0||error
failure|0|abc|error
failure|0|-1|error
CASES
    # The window must hold last Sunday's run as read by this job up to 185 min after its own run was created
    # (mutants timeout 180 + stop 5), and must not reach the run two Sundays back.
    now=$(now_epoch); since=$(since_epoch "$now")
    if [ $((now - 7 * 86400 - 185 * 60)) -ge "$since" ] && [ $((now - 14 * 86400)) -lt "$since" ]; then
        echo "ok    ${WINDOW_DAYS}-day window holds last week's run, not the one before"
    else echo "FAIL  ${WINDOW_DAYS}-day window misses last week's run or reaches two weeks back" >&2; bad=1; fi
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

[ "${1:-}" = "--run" ] || { printf 'usage: %s --run | --self-test\n' "$(basename "$0")" >&2; exit 2; }
: "${RESULT:?}"
[ "$RESULT" = success ] && { echo "mutants: success -- no stop"; exit 0; }
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
API="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/workflows/mutants-nightly.yml"
SINCE=$(since_iso "$(now_epoch)") || exit 1
lrc=0
J=$(curl -sSf -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
      "$API/runs?branch=main&status=success&created=%3E%3D${SINCE}&per_page=1") || lrc=$?
N=$(printf '%s' "${J:-}" | grep -m1 -oE '"total_count": *[0-9]+' | grep -oE '[0-9]+$' || true)
if ! D=$(decide "$RESULT" "$lrc" "$N"); then
    echo "::error::could not count green mutants-nightly runs since $SINCE (listing rc $lrc) -- no stop decision"
    exit 1
fi
case "$D" in
    keep) echo "mutants: $RESULT; ${N:-0} green run(s) since $SINCE -- keeps running" ;;
    stop)
        curl -sSf -X PUT -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$API/disable" \
            || { echo "::error::could not disable mutants-nightly.yml"; exit 1; }
        echo "::warning::C324 7-day stop: mutants $RESULT and no green run since $SINCE -- mutants-nightly.yml DISABLED; owner: gh workflow enable mutants-nightly.yml, then a dispatch" ;;
esac
