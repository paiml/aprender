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
# GitHub reads (GH-1/GH-2): Actions runs and jobs exist only in REST, so they are read with conditional GETs
# (If-None-Match, 304 costs 0) from a cache dir, the listing is paged by Link (100 a page, 10 pages at most), and the job
# reads stop at the first green night: a nightly makes 2 calls when last night was green, never more than the week's runs.
# The cache dir is GUARD_LANE_STOP_CACHE; the workflow restores it across nights (actions/cache). The listing URL has no
# time in it (the window is cut locally, SINCE at UTC midnight), and a finished run's jobs never change, so the job
# reads are 304 from the second night on. The --self-test replays the Link-paging, 304 and window-cut paths on a mock curl.
# An unreadable CLOCK (days live not a number) decides run: the check cannot prove it is past its first week. An
# unreadable HISTORY is the opposite and decides stop, as above.
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

    # --run on a mock curl: Link paging, 304 from the cache, the window cut, an unreadable page
    T=$(mktemp -d "${TMPDIR:-/tmp}/gls-XXXXXX") || exit 2
    trap 'rm -rf "${T:?}"' EXIT
    mkdir "$T/bin"
    cat > "$T/bin/curl" <<'MOCK'
#!/usr/bin/env bash
D="" O="" INM="" U=""
while [ $# -gt 0 ]; do case "$1" in -D) D=$2; shift ;; -o) O=$2; shift ;; -w) shift ;; -H) case "$2" in If-None-Match*) INM=1 ;; esac; shift ;; -*) ;; *) U=$1 ;; esac; shift; done
[ "$MOCK_SCENARIO" = badpage2 ] && case "$U" in *page=2*) exit 7 ;; esac
if [ -n "$INM" ]; then printf 'HTTP/2 304\r\n\r\n' > "$D"; : > "$O"; printf 304; exit 0; fi
green='{"jobs":[{"name":"guards-nightly","steps":[{"name":"ci_guards cargo-free lane self-test (#5920, LAB)","conclusion":"success"}]}]}'
case "$U" in
  *page=3*) printf 'HTTP/2 200\r\netag: "p3"\r\n\r\n' > "$D"; echo '{"workflow_runs":[]}' > "$O" ;;
  *page=2*) printf 'HTTP/2 200\r\netag: "p2"\r\nlink: <%s&page=3>; rel="next"\r\n\r\n' "${U%&page=2}" > "$D"
            if [ "$MOCK_SCENARIO" = oldgreen ]; then echo '{"workflow_runs":[{"id":3,"created_at":"2026-10-01T03:00:00Z"}]}' > "$O"
            else echo '{"workflow_runs":[{"id":2,"created_at":"2026-11-16T03:00:00Z"}]}' > "$O"; fi ;;
  */runs\?*) printf 'HTTP/2 200\r\netag: "p1"\r\nlink: <%s&page=2>; rel="next"\r\n\r\n' "$U" > "$D"
            echo '{"workflow_runs":[{"id":1,"created_at":"2026-11-19T03:00:00Z"}]}' > "$O" ;;
  */runs/1/jobs*) printf 'HTTP/2 200\r\netag: "j1"\r\n\r\n' > "$D"; echo '{"jobs":[{"name":"guards-nightly","steps":[]}]}' > "$O" ;;
  */runs/[23]/jobs*) printf 'HTTP/2 200\r\netag: "j"\r\n\r\n' > "$D"; echo "$green" > "$O" ;;
esac
printf 200
MOCK
    chmod +x "$T/bin/curl"
    # scenario|want decision|pattern in the reason line (each scenario runs twice: the second run reads the cache)
    while IFS='|' read -r sc want pat; do
        rm -rf "${T:?}/cache"; : > "$T/out"
        for _ in 1 2; do
            line=$(env PATH="$T/bin:$PATH" MOCK_SCENARIO="$sc" GH_TOKEN=x GITHUB_REPOSITORY=o/r GITHUB_EVENT_NAME=schedule \
                GITHUB_OUTPUT="$T/out" GUARD_LANE_STOP_CACHE="$T/cache" GUARD_LANE_STOP_NOW=1795000000 bash "$0" --run 2>&1 | tail -n 1)
        done
        if printf '%s\n' "$line" | grep -q "decision $want" && printf '%s\n' "$line" | grep -Eq -- "$pat"; then echo "ok    mock $sc -> $want"
        else printf 'FAIL  mock %s: wanted %s /%s/, got: %s\n' "$sc" "$want" "$pat" "$line" >&2; bad=1; fi
    done <<'MOCKS'
paged|run|5 REST call.*5 not modified
oldgreen|stop|3 REST call.*3 not modified
badpage2|stop|listing unreadable
MOCKS
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

[ "${1:-}" = "--run" ] || { printf 'usage: %s --run | --self-test\n' "$(basename "$0")" >&2; exit 2; }
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_OUTPUT:?}"
emit() { printf 'decision=%s\n' "$1" >> "$GITHUB_OUTPUT"; echo "guard-lane stop: decision $1 ($2)"; }
if [ "${GITHUB_EVENT_NAME:-}" != schedule ]; then emit run "event ${GITHUB_EVENT_NAME:-unset}: a dispatch always runs the check"; exit 0; fi
NOW=${GUARD_LANE_STOP_NOW:-$(date -u +%s)}  # bashrs disable-line=DET002
live=$(( (NOW - $(date -u -d "${ACTIVE_SINCE}T00:00:00Z" +%s)) / 86400 )) || live=
# midnight UTC, so the listing URL below is the same all day and a re-run the same night is a 304
SINCE=$(date -u -d "@$((NOW - WINDOW_DAYS * 86400))" +%Y-%m-%dT00:00:00Z)
API="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions"
auth=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
CACHE="${GUARD_LANE_STOP_CACHE:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/guard-lane-stop-cache}"
mkdir -p "$CACHE" || exit 2
# the counters are files: get runs in a command substitution, so a variable would not survive it
: > "$CACHE/.calls"; : > "$CACHE/.hits"
# get <url> -> prints the path of the body file. GH-1: a conditional GET (If-None-Match) whenever this host has seen the
# URL before, so a 304 costs no allowance and the cached body answers; a fresh runner has no cache and pays one call.
# The headers are cached beside the body so the Link header still pages on a 304.
get() {
    local key body etag hdr code
    key=$(printf '%s' "$1" | sha256sum | cut -c1-16)
    body="$CACHE/$key.body" etag="$CACHE/$key.etag" hdr="$CACHE/$key.hdr"
    local cond=(-H "Cache-Control: no-cache")   # never empty: a bare array expansion trips bashrs SEC002
    if [ -s "$etag" ] && [ -s "$body" ]; then cond=(-H "If-None-Match: $(awk 1 "$etag")"); fi
    echo 1 >> "$CACHE/.calls"
    code=$(curl -sS "${auth[@]}" "${cond[@]}" -D "$hdr.new" -o "$body.new" -w '%{http_code}' "$1") || return 1
    case "$code" in
        200) mv "$body.new" "$body" && mv "$hdr.new" "$hdr"
             sed -n 's/^[Ee][Tt]ag: *//p' "$hdr" | tr -d '\r' | head -n 1 > "$etag" ;;
        304) echo 1 >> "$CACHE/.hits"; rm -f "${body:?}.new" "${hdr:?}.new" ;;
        *) rm -f "${body:?}.new" "${hdr:?}.new"; return 1 ;;
    esac
    printf '%s\n' "$body"
}
# next_url <url> -> the rel="next" page of the URL's cached headers, empty on the last page
next_url() {
    local key
    key=$(printf '%s' "$1" | sha256sum | cut -c1-16)
    sed -n 's/^[Ll]ink:.*<\([^>]*\)>; *rel="next".*/\1/p' "$CACHE/$key.hdr" | head -n 1
}

green=0 why="" pages=0 MAX_PAGES=10
url="$API/workflows/guards-nightly.yml/runs?branch=main&status=completed&per_page=100"
ids=""
# page the listing (100 a page, up to MAX_PAGES), newest first, until a run older than the window: a busy week of dispatches cannot push the nights off the first page
while [ -n "$url" ] && [ "$pages" -lt "$MAX_PAGES" ]; do
    if f=$(get "$url") && page=$(jq -r --arg since "$SINCE" '.workflow_runs[] | select(.created_at >= $since) | .id' "$f") \
        && older=$(jq -r --arg since "$SINCE" '[.workflow_runs[] | select(.created_at < $since)] | length' "$f"); then
        ids="$ids $page"; pages=$((pages + 1)); url=$(next_url "$url")
        [ "$older" -eq 0 ] || url=""   # newest first: a run older than the window ends the paging
    else
        why=" run listing unreadable (counted as no green);"; ids=""; break
    fi
done
[ -z "$url" ] || [ -n "$why" ] || why=" run listing capped at $MAX_PAGES pages;"
for id in $ids; do
    # a run whose jobs cannot be read, or that lacks the step, adds no green; one green night is the whole answer
    if f=$(get "$API/runs/$id/jobs?per_page=100") && c=$(jq -r --arg j "$JOB_NAME" --arg s "$STEP_NAME" \
            '.jobs[] | select(.name == $j) | .steps[] | select(.name == $s) | .conclusion' "$f"); then
        if [ "$c" = success ]; then green=1; break; fi
    else
        why="$why run $id unreadable;"
    fi
done
D=$(decide "$live" "$green")
calls=$(wc -l < "$CACHE/.calls") hits=$(wc -l < "$CACHE/.hits")
emit "$D" "live ${live:-?} day(s), $green green night(s) since $SINCE, ${calls// /} REST call(s) (${hits// /} not modified);$why"
