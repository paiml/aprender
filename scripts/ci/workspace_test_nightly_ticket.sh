#!/usr/bin/env bash
# workspace_test_nightly_ticket.sh -- the workspace-test nightly's one ticket and its 7-day stop (#5002).
#
# LAB (operator C324 5): "A red opens or updates one ticket with one owner, then stays silent. Red
# or unmeasured for 7 days, it stops running until its owner brings it back." The nightly
# (.github/workflows/workspace-test-nightly.yml) holds no credential. This script runs in
# .github/workflows/workspace-test-nightly-ticket.yml, which holds issues: write and actions: write.
# NO PYTHON (operator C301): bash + jq.
#
#   ticket CONCLUSION SHA URL   a night on main that is not green: open the one ticket, or comment
#                               on it once per commit. failure -> red; any other conclusion that is
#                               not success, skipped or neutral -> not_measured, also recorded.
#   stale                       once a day: when the nightly has been enabled for 7 days and main has
#                               had no green run of it in the last 7 days (no run at all counts),
#                               say so on the ticket, then disable the nightly. Its owner re-enables it.
#   --self-test                 every case through a gh stub, and the wiring of both workflows.
#
# A read or write that fails is NOT-MEASURED and exit 1: never a ticket, never a stop.
# WTN_GH (a gh stub) and WTN_NOW (epoch seconds) are for the self-test only.
#
# EXIT 0 done, or nothing to do · 1 a read or write failed · 2 usage
set -euo pipefail

PROG=workspace_test_nightly_ticket.sh
SELF=$(readlink -f "${BASH_SOURCE[0]}")
WF=workspace-test-nightly.yml
TITLE="workspace-test nightly: not green"
OWNER=aprender-a7n
DAY=86400
PAGE=50

die() { echo "$PROG: $*" >&2; exit 2; }
need_jq() { command -v jq > /dev/null 2>&1 || { echo "NOT-MEASURED: jq not found"; return 1; }; }

# upsert MARK TAG BODY: on the one open ticket (the lowest number with the exact title), comment
# BODY unless the ticket already names MARK; with no ticket, open it with BODY. A search that
# fills its page, or output that is not a JSON list, proves nothing: no ticket is opened then.
upsert() {
    local mark=$1 tag=$2 body=$3 gh=${WTN_GH:-gh} j n len
    if ! j=$("$gh" issue list --state open --limit "$PAGE" --search "\"$TITLE\" in:title" --json number,title); then
        echo "NOT-MEASURED: the issue search for '$TITLE' failed"; return 1
    fi
    n=$(jq -r --arg t "$TITLE" '[.[] | select(.title == $t) | .number] | min // empty' <<< "$j" 2> /dev/null) || n=""
    if [ -z "$n" ]; then
        len=$(jq 'if type == "array" then length else error end' <<< "$j" 2> /dev/null) || len=""
        if ! [[ $len =~ ^[0-9]+$ ]] || [ "$len" -ge "$PAGE" ]; then
            echo "NOT-MEASURED: the issue search for '$TITLE' returned a full page or no readable list"; return 1
        fi
    fi
    if [ -n "$n" ]; then
        if ! j=$("$gh" issue view "$n" --json body,comments); then echo "NOT-MEASURED: reading #$n failed"; return 1; fi
        if grep -qF -- "$mark" <<< "$j"; then echo "TICKET kept #$n (already names $mark)"
        elif "$gh" issue comment "$n" --body "$body" > /dev/null; then echo "TICKET updated #$n: $tag"
        else echo "NOT-MEASURED: commenting on #$n failed"; return 1; fi
    else
        if ! n=$("$gh" issue create --title "$TITLE" --label "owner:$OWNER" --body "$body"); then
            echo "NOT-MEASURED: opening the ticket failed"; return 1
        fi
        n=${n##*/}
        [[ $n =~ ^[0-9]+$ ]] || { echo "NOT-MEASURED: opening the ticket printed no issue url"; return 1; }
        echo "TICKET opened #$n: $tag"
    fi
}

ticket() {
    local concl=$1 c=$2 url=$3 state mark
    [[ $c =~ ^[0-9a-f]{40}$ ]] || die "ticket: '$c' is not a commit sha"
    case "$concl" in
        success | skipped | neutral) echo "TICKET none: the nightly at ${c:0:9} concluded $concl"; return 0 ;;
        failure) state=red ;;
        ?*) state=not_measured ;;
        *) die "ticket: empty conclusion" ;;
    esac
    need_jq || return 1
    mark="workspace-test-nightly@$c"
    upsert "$mark" "$state" "The workspace-test nightly at ${c:0:9} is $state ($concl): $url. It runs main's full workspace tests (#5002) and is a lab check: it cannot block a merge or a release. A green night does not close this ticket; its owner does, once the nightly is green again. With no green run on main for 7 days the nightly stops itself and says so here. Owner: $OWNER. $mark"
}

stale() {
    local gh=${WTN_GH:-gh} now=${WTN_NOW:-} j state upd anchor since n total mark
    need_jq || return 1
    [ -n "$now" ] || now=$(date -u +%s)
    [[ $now =~ ^[0-9]+$ ]] || die "stale: WTN_NOW '$now' is not epoch seconds"
    if ! j=$("$gh" api "repos/{owner}/{repo}/actions/workflows/$WF"); then echo "NOT-MEASURED: reading the workflow $WF failed"; return 1; fi
    state=$(jq -r '.state // empty' <<< "$j" 2> /dev/null) || state=""
    upd=$(jq -r '.updated_at // empty' <<< "$j" 2> /dev/null) || upd=""
    [ -n "$state" ] || { echo "NOT-MEASURED: the workflow $WF has no readable state"; return 1; }
    if [ "$state" != active ]; then echo "STALE already stopped: $WF is $state; its owner re-enables it"; return 0; fi
    # updated_at moves when the workflow is enabled or disabled, so the 7 days count from the
    # owner's last re-enable (or the merge that added it), never from before it last ran.
    # An empty string is a valid `date -d` input (midnight today), so it is refused before.
    anchor=""; [ -z "$upd" ] || anchor=$(date -u -d "$upd" +%s 2> /dev/null) || anchor=""
    [[ $anchor =~ ^[0-9]+$ ]] || { echo "NOT-MEASURED: the workflow $WF has no readable updated_at ('$upd')"; return 1; }
    if [ $((now - anchor)) -lt $((7 * DAY)) ]; then echo "STALE no: $WF has been enabled for under 7 days (since $upd)"; return 0; fi
    since=$(date -u -d "@$((now - 7 * DAY))" +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002 (C324 5: the 7-day window IS wall-clock)
    if ! j=$("$gh" api "repos/{owner}/{repo}/actions/workflows/$WF/runs?branch=main&status=success&per_page=100&created=%3E%3D$since"); then
        echo "NOT-MEASURED: reading the green runs of $WF on main failed"; return 1
    fi
    total=$(jq -r '.total_count' <<< "$j" 2> /dev/null) || total=""
    n=$(jq -r --arg s "$since" '[.workflow_runs[] | select(.conclusion == "success" and .head_branch == "main" and .created_at >= $s)] | length' <<< "$j" 2> /dev/null) || n=""
    [[ $total =~ ^[0-9]+$ && $n =~ ^[0-9]+$ ]] || { echo "NOT-MEASURED: the run list of $WF is not readable"; return 1; }
    if [ "$n" -gt 0 ]; then echo "STALE no: $n green run(s) of $WF on main since $since"; return 0; fi
    # The server filtered for green runs on main; runs it returned that this cannot confirm are
    # not evidence either way.
    [ "$total" = 0 ] || { echo "NOT-MEASURED: the run list holds $total run(s) not confirmed green on main since $since"; return 1; }
    # The ticket first: a stop that is not on the ticket would be silent, and the next day's run
    # would find the workflow disabled and say nothing. The mark is the anchor, so each re-enable
    # gets its own note.
    mark="workspace-test-nightly-stopped@$upd"
    upsert "$mark" stopped "The workspace-test nightly has had no green run on main since $since (7 days; it has been enabled since $upd), so it stops itself now. Its owner brings it back with \`gh workflow enable $WF\`. Owner: $OWNER. $mark" || return 1
    "$gh" workflow disable "$WF" > /dev/null || { echo "NOT-MEASURED: disabling $WF failed"; return 1; }
    echo "STALE stopped: $WF disabled (no green run on main since $since)"
}

# ---------------------------------------------------------------------------------------------
PASS=0; FAIL=0
row() { # row NAME WANT_RC GOT_RC WANT_PATTERN OUTPUT
    if [ "$2" = "$3" ] && grep -q -E -- "$4" <<< "$5"; then PASS=$((PASS + 1)); printf 'PASS  %s\n' "$1"
    else FAIL=$((FAIL + 1)); printf 'FAIL  %s (want rc=%s /%s/, got rc=%s)\n%s\n' "$1" "$2" "$4" "$3" "$5"; fi
}

self_test() {
    # d is global: the EXIT trap runs after this function returns, when a local d is unset.
    local sha=0123456789abcdef0123456789abcdef01234567 t=$TITLE out now since name
    # The rows read each case's exit status; the cases themselves run as child processes, under
    # the script's own set -euo pipefail, exactly as the workflow runs them.
    set +e
    need_jq || return 1
    d=$(mktemp -d)
    trap 'rm -rf "${d:?}"' EXIT
    cat > "$d/gh" << 'GH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FXGH_LOG"
f=${FXGH_FAIL:-}
case "$1" in
    api) case "$2" in
            */runs*) [ "$f" != runs ] || exit 1; cat -- "$FXGH_RUNS" ;;
            *) [ "$f" != wf ] || exit 1; cat -- "$FXGH_WF" ;;
         esac ;;
    issue) case "$2" in
            list) [ "$f" != list ] || exit 1; cat -- "$FXGH_LIST" ;;
            view) [ "$f" != view ] || exit 1; cat -- "$FXGH_VIEW" ;;
            comment) [ "$f" != comment ] || exit 1 ;;
            create) [ "$f" != create ] || exit 1; echo "${FXGH_CREATED:-https://github.invalid/o/r/issues/77}" ;;
            *) exit 9 ;;
           esac ;;
    workflow) [ "$2" = disable ] || exit 9; [ "$f" != disable ] || exit 1 ;;
    *) exit 9 ;;
esac
GH
    chmod +x "$d/gh"
    # gh_run LIST VIEW WF RUNS FAIL ARGS...: this script with ARGS, through the stub; then one GH
    # line of the calls it made, in order.
    gh_run() {
        printf '%s\n' "$1" > "$d/list"; printf '%s\n' "$2" > "$d/view"
        printf '%s\n' "$3" > "$d/wf"; printf '%s\n' "$4" > "$d/runs"; : > "$d/log"
        local fl=$5 rc=0; shift 5
        env WTN_GH="$d/gh" FXGH_LOG="$d/log" FXGH_LIST="$d/list" FXGH_VIEW="$d/view" \
            FXGH_WF="$d/wf" FXGH_RUNS="$d/runs" FXGH_FAIL="$fl" bash "$SELF" "$@" || rc=$?
        printf 'GH %s\n' "$(tr '\n' ';' < "$d/log")"; return "$rc"
    }
    tk() { gh_run "$2" "$3" '{}' '{}' "${4:-}" ticket "$1" "$sha" https://github.invalid/run/1; }

    echo "self-test: ticket"
    out=$(tk success '[]' '{}'); row "ticket: a green night opens nothing" 0 $? '^TICKET none.*GH *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(tk failure '[]' '{}'); row "ticket: a red night, no ticket -> opens one with its owner" 0 $? "TICKET opened #77: red.*issue create --title $t --label owner:aprender-a7n --body .*is red \(failure\).*Owner: aprender-a7n\. workspace-test-nightly@$sha" "$(tr '\n' ' ' <<< "$out")"
    out=$(tk cancelled '[]' '{}'); row "ticket: a cancelled night -> a not_measured ticket" 0 $? 'TICKET opened #77: not_measured' "$out"
    out=$(tk failure "[{\"number\":12,\"title\":\"$t\"},{\"number\":9,\"title\":\"$t\"}]" '{"body":"x"}')
    row "ticket: an open ticket -> one comment on the lowest" 0 $? 'TICKET updated #9: red.*issue view 9 .*issue comment 9 ' "$(tr '\n' ' ' <<< "$out")"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t\"}]" "{\"body\":\"workspace-test-nightly@$sha\"}")
    row "ticket: the commit already named -> kept" 0 $? 'TICKET kept #9' "$out"
    grep -q 'issue comment' <<< "$out"; row "ticket: kept means no comment call" 1 $? '.' "$out"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t (old)\"}]" '{}'); row "ticket: a title that only contains it -> opens its own" 0 $? 'TICKET opened #77' "$out"
    out=$(tk failure '[]' '{}' list); row "ticket: the search fails -> not_measured" 1 $? 'NOT-MEASURED: the issue search' "$out"
    grep -q 'issue create' <<< "$out"; row "ticket: a failed search opens nothing" 1 $? '.' "$out"
    out=$(tk failure "$(jq -cn '[range(50) | {number: (. + 100), title: "other"}]')" '{}'); row "ticket: a full page without the title -> opens nothing" 1 $? 'full page or no readable list' "$out"
    out=$(tk failure '{"not":"a list"}' '{}'); row "ticket: a search that is not a list -> opens nothing" 1 $? 'full page or no readable list' "$out"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t\"}]" '{}' view); row "ticket: reading the ticket fails -> fails" 1 $? 'NOT-MEASURED: reading #9' "$out"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t\"}]" '{"body":"x"}' comment); row "ticket: the comment fails -> fails" 1 $? 'NOT-MEASURED: commenting on #9' "$out"
    out=$(tk failure '[]' '{}' create); row "ticket: opening fails -> fails" 1 $? 'NOT-MEASURED: opening the ticket failed' "$out"
    out=$(FXGH_CREATED=oops tk failure '[]' '{}'); row "ticket: opening prints no url -> fails" 1 $? 'printed no issue url' "$out"
    out=$(gh_run '[]' '{}' '{}' '{}' '' ticket failure notasha https://github.invalid/run/1 2>&1); row "ticket: a bad sha is usage, no gh call" 2 $? "not a commit sha.*GH *$" "$(tr '\n' ' ' <<< "$out")"

    echo "self-test: stale"
    now=1792454400; since=2026-10-13T00:00:00Z  # now is 2026-10-20T00:00:00Z
    local old='{"state":"active","updated_at":"2026-09-01T10:00:00.000+02:00"}' none='{"total_count":0,"workflow_runs":[]}'
    local open9="[{\"number\":9,\"title\":\"$t\"}]"
    st() { WTN_NOW=$now gh_run "$1" "$2" "$3" "$4" "${5:-}" stale; }
    out=$(st '[]' '{}' '{"state":"disabled_manually","updated_at":"2026-10-01T00:00:00Z"}' "$none")
    row "stale: already disabled -> no write" 0 $? 'STALE already stopped: .* is disabled_manually.*GH api [^;]*nightly\.yml; *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(st '[]' '{}' '{"state":"active","updated_at":"2026-10-18T10:00:00.000+02:00"}' "$none")
    row "stale: enabled under 7 days ago -> no run read, no write" 0 $? 'STALE no: .*under 7 days.*GH api [^;]*nightly\.yml; *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(st '[]' '{}' '{"state":"active","updated_at":"2026-10-13T00:00:00Z"}' "$none")
    row "stale: enabled exactly 7 days ago -> the runs are read" 0 $? 'STALE stopped' "$out"
    out=$(st '[]' '{}' "$old" '{"total_count":1,"workflow_runs":[{"conclusion":"success","head_branch":"main","created_at":"2026-10-15T03:00:00Z"}]}')
    row "stale: a green main run in 7 days -> no write" 0 $? 'STALE no: 1 green.*GH api [^;]*nightly\.yml;api [^;]*runs[^;]*; *$' "$(tr '\n' ' ' <<< "$out")"
    grep -qF "runs?branch=main&status=success&per_page=100&created=%3E%3D$since;" <<< "$out"
    row "stale: the run query asks for green main runs since now - 7 days" 0 $? '.' "$out"
    out=$(st "$open9" '{"body":"x"}' "$old" "$none")
    row "stale: no green run -> a note on the ticket, then disable" 0 $? 'TICKET updated #9: stopped.*STALE stopped.*issue comment 9 --body [^;]*no green run on main since 2026-10-13T00:00:00Z.*Owner: aprender-a7n\. workspace-test-nightly-stopped@2026-09-01T10:00:00\.000\+02:00;workflow disable workspace-test-nightly\.yml; *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(st '[]' '{}' "$old" "$none")
    row "stale: no green run, no ticket -> opens it, then disable" 0 $? 'TICKET opened #77: stopped.*STALE stopped.*issue create .*workflow disable' "$(tr '\n' ' ' <<< "$out")"
    out=$(st "$open9" '{"body":"workspace-test-nightly-stopped@2026-09-01T10:00:00.000+02:00"}' "$old" "$none")
    row "stale: the stop already noted -> no comment, disable again" 0 $? 'TICKET kept #9.*STALE stopped' "$(tr '\n' ' ' <<< "$out")"
    grep -q 'issue comment' <<< "$out"; row "stale: noted means no second comment" 1 $? '.' "$out"
    out=$(st '[]' '{}' "$old" "$none" wf); row "stale: the workflow read fails -> not_measured, no write" 1 $? 'NOT-MEASURED: reading the workflow.*GH api [^;]*; *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(st '[]' '{}' '{"updated_at":"2026-09-01T10:00:00Z"}' "$none"); row "stale: no state -> not_measured" 1 $? 'no readable state' "$out"
    out=$(st '[]' '{}' '{"state":"active","updated_at":"soon"}' "$none"); row "stale: an unreadable updated_at -> not_measured" 1 $? "no readable updated_at \('soon'\)" "$out"
    out=$(st '[]' '{}' '{"state":"active"}' "$none"); row "stale: no updated_at is not 'today' -> not_measured" 1 $? "no readable updated_at \(''\)" "$out"
    out=$(st '[]' '{}' "$old" "$none" runs); row "stale: the run read fails -> not_measured, no write" 1 $? 'NOT-MEASURED: reading the green runs.*GH api [^;]*;api [^;]*; *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(st '[]' '{}' "$old" '{}'); row "stale: a run list that is not readable -> not_measured" 1 $? 'run list of .* is not readable' "$out"
    out=$(st '[]' '{}' "$old" '{"total_count":1,"workflow_runs":[{"conclusion":"success","head_branch":"proof","created_at":"2026-10-15T03:00:00Z"}]}')
    row "stale: a green run on another branch is not main's -> not_measured, no write" 1 $? 'not confirmed green on main.*GH api [^;]*;api [^;]*; *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(st '[]' '{}' "$old" '{"total_count":1,"workflow_runs":[{"conclusion":"success","head_branch":"main","created_at":"2026-10-12T23:59:59Z"}]}')
    row "stale: a green run older than 7 days is not evidence -> not_measured" 1 $? 'not confirmed green on main' "$out"
    out=$(st "$open9" '{"body":"x"}' "$old" "$none" comment); row "stale: the note fails -> not_measured, never disabled" 1 $? 'NOT-MEASURED: commenting on #9' "$out"
    grep -q 'workflow disable' <<< "$out"; row "stale: a failed note means no disable call" 1 $? '.' "$out"
    out=$(st "$open9" '{"body":"x"}' "$old" "$none" disable); row "stale: the disable fails -> not_measured" 1 $? 'NOT-MEASURED: disabling' "$out"

    echo "self-test: wiring"
    name=$(sed -n 's/^name: //p' ".github/workflows/$WF")
    [ -n "$name" ]; row "wiring: the nightly workflow $WF has a name" 0 $? '.' "$name"
    grep -qF "workflows: [\"$name\"]" .github/workflows/workspace-test-nightly-ticket.yml
    row "wiring: the ticket workflow listens to '$name'" 0 $? '.' "$name"
    grep -qF "scripts/ci/$PROG ticket \"\$CONCLUSION\" \"\$RUN_SHA\" \"\$RUN_URL\"" .github/workflows/workspace-test-nightly-ticket.yml
    row "wiring: the ticket workflow runs ticket" 0 $? '.' x
    grep -qF "scripts/ci/$PROG stale" .github/workflows/workspace-test-nightly-ticket.yml
    row "wiring: the ticket workflow runs stale" 0 $? '.' x

    printf '%s self-test: %d pass, %d fail\n' "$PROG" "$PASS" "$FAIL"
    [ "$FAIL" -eq 0 ]
}

case "${1:-}" in
    ticket) [ $# -eq 4 ] || die "usage: ticket CONCLUSION SHA URL"; ticket "$2" "$3" "$4" ;;
    stale) [ $# -eq 1 ] || die "usage: stale"; stale ;;
    --self-test) self_test ;;
    *) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' >&2; exit 2 ;;
esac
