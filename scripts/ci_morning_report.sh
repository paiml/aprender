#!/usr/bin/env bash
# ci_morning_report.sh — the merge queue's morning numbers, every red tagged by machine.
#
#   ci_morning_report.sh --self-test
#   ci_morning_report.sh [--days N] [--end YYYY-MM-DD] [--repo OWNER/NAME] [--lessons FILE] [--tsv FILE]
#
# Window: the N whole UTC days that end before --end (default: today, UTC), so the print does not change
# while the day runs. For every merge_group run of ci.yml in the window it prints
#   queue runs, pass rate, wall kills, repeats of a cause that already has a lesson,
#   host-hours lost to runs that failed on good code, merges per day,
# and the "mastered" bar: N >= 7 days with runs, >= 50 queue runs, >= 15 merges EVERY day, pass rate >= 90%,
# wall kills 0, repeats 0.
#
# Tagging is mechanical, never read by eye. A red's cause key is <job>/<section>[/<guard step>]:
#   - section results come from the gate job's X86:/DET: lines, the per-section results.json output
#     (fat_driver emit_results_output) as the gate job received it; source=results
#   - when the gate job did not run, from the "section S: failure" annotations; source=annotations
#   - a non-sectioned job (workspace-test-shard, mac-check, ...) is keyed by its job name; source=jobs
#   - wall kill = the "step exceeded its timeout" annotation (fat_driver spawn: a step killed at its deadline)
#     or a job cancelled at its maximum execution time. An external-job wait message ("not completed within
#     Ns") is NOT one: it is also printed on 65-minute jobs whose dependency was cancelled.
#   - a sectioned job with no section result is keyed by the runner's own failure annotation.
# Repeat: a red whose cause key matches a row of the lesson file (default ci/red-lessons.tsv:
# <cause glob> TAB <lesson ref> TAB <lesson date YYYY-MM-DD>) and that started after the lesson's date.
# Good code: the PR merged later, and its squash commit carries the same patch (git patch-id --stable) as the
# failed queue commit, so the code that failed is the code that landed. Host-hours = sum of its jobs' run time.
#
# A number that could not be read prints not_measured, and the mastered bar is then NO: a skip is never a pass.
# GitHub: one GET per object; completed runs are immutable and cached for good, list pages use ETag (304 is
# free). The calls made are printed. Exit 0 = report printed, 1 = a read failed (report still printed with
# not_measured), 2 = usage.
set -uo pipefail

die() { echo "ci_morning_report: $*" >&2; exit 2; }

# --self-test: a fake `gh` serves fixed API answers and a scratch git repo holds the queue commits, then the
# report is run for real (a child process, no flag inherited) and every number is checked both ways.
self_test() {
    local t fix bin repo out tsv b q2 m2 q3 m3 fails=0
    t=$(mktemp -d) || die "mktemp failed"
    # shellcheck disable=SC2064  # expand $t now
    trap "rm -rf -- '${t:?}'" RETURN
    fix="$t/fix" bin="$t/bin" repo="$t/repo"
    mkdir -p "$fix" "$bin" "$repo" || return 1
    cat > "$bin/gh" << 'GH'
#!/usr/bin/env bash
inc=0 url=""
while [ "$#" -gt 0 ]; do case "$1" in api) shift ;; -i) inc=1; shift ;; -H) shift 2 ;; *) url=$1; shift ;; esac; done
f="$CMR_FIX/$(printf '%s' "$url" | sha256sum | cut -c1-40)"
[ -f "$f" ] || exit 1
[ "$inc" = 1 ] && printf 'HTTP/2.0 200 OK\r\nEtag: W/"t"\r\n\r\n'
cat "$f"
GH
    chmod +x "$bin/gh"
    put() { printf '%s\n' "$2" > "$fix/$(printf '%s' "$1" | sha256sum | cut -c1-40)"; }
    # queue commits: q2 and m2 carry the same change on different bases (good code); q3 and m3 differ
    git -C "$repo" init -q && git -C "$repo" config user.email t@t && git -C "$repo" config user.name t
    commit() { printf '%s\n' "$2" > "$repo/$1" && git -C "$repo" add -A && git -C "$repo" -c core.hooksPath=/dev/null commit -qm "$1" && git -C "$repo" rev-parse HEAD; }
    b=$(commit a base) || return 1
    q2=$(commit f2 x) || return 1
    git -C "$repo" checkout -q "$b" && commit c other > /dev/null && m2=$(commit f2 x) || return 1
    git -C "$repo" checkout -q "$b" && q3=$(commit f3 y) || return 1
    git -C "$repo" checkout -q "$b" && m3=$(commit f3 z) || return 1
    local R=repos/o/r
    put "$R/actions/workflows/ci.yml/runs?event=merge_group&created=2026-01-01..2026-01-03&per_page=100&page=1" '{"workflow_runs":[
      {"id":11,"head_branch":"gh-readonly-queue/main/pr-1-b","head_sha":"x","status":"completed","conclusion":"success","run_started_at":"2026-01-01T01:00:00Z","updated_at":"","created_at":"2026-01-01T01:00:00Z"},
      {"id":12,"head_branch":"gh-readonly-queue/main/pr-2-b","head_sha":"'"$q2"'","status":"completed","conclusion":"failure","run_started_at":"2026-01-02T10:00:00Z","updated_at":"","created_at":"2026-01-02T10:00:00Z"},
      {"id":13,"head_branch":"gh-readonly-queue/main/pr-3-b","head_sha":"'"$q3"'","status":"completed","conclusion":"failure","run_started_at":"2026-01-03T10:00:00Z","updated_at":"","created_at":"2026-01-03T10:00:00Z"},
      {"id":14,"head_branch":"gh-readonly-queue/main/pr-4-b","head_sha":"x","status":"completed","conclusion":"cancelled","run_started_at":"2026-01-03T11:00:00Z","updated_at":"","created_at":"2026-01-03T11:00:00Z"},
      {"id":15,"head_branch":"gh-readonly-queue/main/pr-5-b","head_sha":"x","status":"in_progress","conclusion":null,"run_started_at":"2026-01-03T12:00:00Z","updated_at":"","created_at":"2026-01-03T12:00:00Z"}]}'
    put "$R/actions/runs/12/jobs?per_page=100" '{"jobs":[
      {"id":120,"name":"x86-main","conclusion":"failure","started_at":"2026-01-02T10:00:00Z","completed_at":"2026-01-02T11:00:00Z"},
      {"id":121,"name":"gate","conclusion":"failure","started_at":"2026-01-02T11:00:00Z","completed_at":"2026-01-02T11:00:36Z"},
      {"id":122,"name":"yoga","conclusion":"skipped","started_at":"2026-01-02T10:00:00Z","completed_at":"2026-01-02T15:00:00Z"}]}'
    put "$R/check-runs/120/annotations?per_page=100" '[{"annotation_level":"failure","title":"","message":"step exceeded its timeout (3570s); killing group 1"},
      {"annotation_level":"failure","title":"","message":"section guard-cargo: failure"},
      {"annotation_level":"warning","title":"","message":"step exceeded its timeout (1s); killing group 2"}]'
    put "$R/check-runs/121/annotations?per_page=100" '[]'
    put "$R/actions/jobs/121/logs" '2026-01-02T11:00:01.0Z   X86: {"guard-cargo":{"result":"failure","continue_on_error":false,"outputs":{}},"sov.lint":{"result":"failure","continue_on_error":true,"outputs":{}},"sov.test":{"result":"success","continue_on_error":false,"outputs":{}}}'
    put "$R/actions/runs/13/jobs?per_page=100" '{"jobs":[
      {"id":130,"name":"mac-check","conclusion":"failure","started_at":"2026-01-03T10:00:00Z","completed_at":"2026-01-03T10:30:00Z"},
      {"id":131,"name":"workspace-test","conclusion":"failure","started_at":"2026-01-03T10:00:00Z","completed_at":"2026-01-03T10:01:00Z"},
      {"id":132,"name":"gate","conclusion":"skipped","started_at":null,"completed_at":null}]}'
    put "$R/check-runs/130/annotations?per_page=100" '[{"annotation_level":"failure","title":"","message":"Process completed with exit code 1."}]'
    put "$R/check-runs/131/annotations?per_page=100" '[]'
    put "$R/actions/runs/14/jobs?per_page=100" '{"jobs":[{"id":140,"name":"x86-main","conclusion":"cancelled","started_at":"2026-01-03T11:00:00Z","completed_at":"2026-01-03T11:05:00Z"}]}'
    put "$R/check-runs/140/annotations?per_page=100" '[{"annotation_level":"failure","title":"","message":"The self-hosted runner lost communication with the server. Verify the machine is running."},
      {"annotation_level":"failure","title":"","message":"external job '"'"'workspace-test'"'"': not completed within 12000s"}]'
    put "$R/pulls/2" '{"merged_at":"2026-01-02T12:00:00Z","merge_commit_sha":"'"$m2"'"}'
    put "$R/pulls/3" '{"merged_at":"2026-01-03T12:00:00Z","merge_commit_sha":"'"$m3"'"}'
    put "search/issues?q=repo:o/r+is:pr+is:merged+merged:2026-01-01&per_page=1" '{"total_count":20}'
    put "search/issues?q=repo:o/r+is:pr+is:merged+merged:2026-01-02&per_page=1" '{"total_count":14}'
    put "search/issues?q=repo:o/r+is:pr+is:merged+merged:2026-01-03&per_page=1" '{"total_count":16}'
    printf '# lessons\nmac-check\tL-1\t2026-01-02\nx86-main/guard-cargo wallkill:*\tL-2\t2026-01-05\n' > "$t/lessons.tsv"
    tsv="$t/reds.tsv"
    out=$(cd "$repo" && PATH="$bin:$PATH" CMR_FIX="$fix" XDG_CACHE_HOME="$t/cache" \
        bash "$SELF" --repo o/r --days 3 --end 2026-01-04 --lessons "$t/lessons.tsv" --tsv "$tsv" 2>&1)
    local rc=$?
    want() { # want <must-hit|must-not-hit> <haystack name> <haystack> <fixed string>
        local hit=0
        grep -qF -- "$4" <<< "$3" && hit=1
        if { [ "$1" = must-hit ] && [ "$hit" = 1 ]; } || { [ "$1" = must-not-hit ] && [ "$hit" = 0 ]; }; then
            printf 'ok    %-12s %s: %s\n' "$1" "$2" "$4"
        else printf 'FAIL  %-12s %s: %s\n' "$1" "$2" "$4"; fails=$((fails + 1)); fi
    }
    local rows; rows=$(cat "$tsv" 2> /dev/null)
    [ "$rc" = 0 ] && echo "ok    exit 0" || { echo "FAIL  exit $rc"; fails=$((fails + 1)); }
    want must-hit report "$out" "queue runs        4  (success 1, failure 2, cancelled 1; 1 still running"
    want must-hit report "$out" "pass rate         33.3%"
    want must-hit report "$out" "wall kills        1 runs"
    want must-hit report "$out" "repeats           1  (lesson file lessons.tsv: 2 rows)"
    want must-hit report "$out" "  13 mac-check -> L-1"
    want must-hit report "$out" "host-hours lost   1.0 h on 1 runs that failed on good code (0 not decidable)"
    want must-hit report "$out" "merges/day        min 14, mean 16.7"
    want must-hit report "$out" "mastered: NO  days 3/7 FAIL; runs 4/50 FAIL; merges min 14/15 FAIL; pass 33.3%/90 FAIL; wall kills 1/0 FAIL; repeats 1/0 FAIL"
    want must-hit reds "$rows" $'12\t2026-01-02\t2\tx86-main/guard-cargo\tstep\tresults'
    want must-hit reds "$rows" $'13\t2026-01-03\t3\tmac-check\t0\tjobs'
    want must-hit reds "$rows" $'14\t2026-01-03\t4\tx86-main/The self-hosted runner lost communication with the server\t0\tannotations'
    want must-not-hit reds "$rows" "sov.lint"
    want must-not-hit reds "$rows" "workspace-test"
    want must-not-hit reds "$rows" "untagged"
    want must-not-hit report "$out" "x86-main/guard-cargo -> L-2"
    echo "self-test: $fails failed"
    [ "$fails" = 0 ]
}

SELF=$(realpath -- "$0")
if [ "${1:-}" = --self-test ]; then self_test; exit $?; fi
DAYS=7 END="" REPO="paiml/aprender" LESSONS="" TSV=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --days) [ "$#" -ge 2 ] || die "--days needs a value"; DAYS=$2; shift 2 ;;
        --end) [ "$#" -ge 2 ] || die "--end needs a value"; END=$2; shift 2 ;;
        --repo) [ "$#" -ge 2 ] || die "--repo needs a value"; REPO=$2; shift 2 ;;
        --lessons) [ "$#" -ge 2 ] || die "--lessons needs a value"; LESSONS=$2; shift 2 ;;
        --tsv) [ "$#" -ge 2 ] || die "--tsv needs a value"; TSV=$2; shift 2 ;;
        -h | --help) sed -n '2,30p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ $DAYS =~ ^[1-9][0-9]?$ ]] || die "--days wants 1..99, got '$DAYS'"
[ -n "$END" ] || END=$(date -u +%F)
[[ $END =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--end wants YYYY-MM-DD, got '$END'"
START=$(date -u -d "$END - $DAYS days" +%F) || die "bad --end '$END'"  # bashrs disable-line=DET002 (calendar arithmetic on --end, no clock read)
LAST=$(date -u -d "$END - 1 day" +%F) || die "bad --end '$END'"  # bashrs disable-line=DET002 (calendar arithmetic on --end, no clock read)
ROOT=$(git rev-parse --show-toplevel 2> /dev/null) || die "run inside a clone of $REPO: good-code checks read git objects"
[ -n "$LESSONS" ] || LESSONS="$ROOT/ci/red-lessons.tsv"
LESSONS=$(realpath -m -- "$LESSONS"); [ -z "$TSV" ] || TSV=$(realpath -m -- "$TSV")
cd "$ROOT" || die "cannot cd to $ROOT"
command -v gh > /dev/null || die "gh not on PATH"
command -v jq > /dev/null || die "jq not on PATH"

CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/ci-morning-report"
mkdir -p "$CACHE" || die "cannot create $CACHE"
LEDGER=$(mktemp) || die "mktemp failed"   # call ledger: the fetchers run in $(...) subshells

key() { printf '%s' "$1" | sha256sum | cut -c1-40; }

# get_fixed <api path> — an object that never changes once its run completed. Cached for good.
get_fixed() {
    local k f
    k=$(key "$1"); f="$CACHE/$k"
    if [ -s "$f" ]; then echo hit >> "$LEDGER"; cat "$f"; return 0; fi
    echo call >> "$LEDGER"
    if gh api "$1" > "$f.tmp" 2> /dev/null; then mv -f "$f.tmp" "$f"; cat "$f"; return 0; fi
    rm -f -- "${f:?}.tmp"; echo fail >> "$LEDGER"; return 1
}

# get_etag <api path> — a list that can change; If-None-Match makes an unchanged read free.
get_etag() {
    local k f hdr etag="" code
    k=$(key "$1"); f="$CACHE/$k"
    [ -s "$f.etag" ] && [ -s "$f" ] && etag=$(cat "$f.etag")
    echo call >> "$LEDGER"
    gh api -i ${etag:+-H "If-None-Match: $etag"} "$1" > "$f.raw" 2> /dev/null
    code=$(head -1 "$f.raw" | awk '{print $2}')
    if [ "$code" = 304 ]; then echo hit >> "$LEDGER"; rm -f -- "${f:?}.raw"; cat "$f"; return 0; fi
    if [ "$code" != 200 ]; then rm -f -- "${f:?}.raw"; echo fail >> "$LEDGER"; return 1; fi
    hdr=$(sed -n '1,/^\r\{0,1\}$/p' "$f.raw")
    sed '1,/^\r\{0,1\}$/d' "$f.raw" > "$f"
    printf '%s\n' "$hdr" | awk 'tolower($1)=="etag:" {sub(/\r$/,""); $1=""; sub(/^ /,""); print}' > "$f.etag"
    rm -f -- "${f:?}.raw"
    cat "$f"
}

# ---- 1. queue runs in the window ------------------------------------------------------------------
RUNS=$(mktemp) || die "mktemp failed"
REDS=$(mktemp) || die "mktemp failed"
trap 'rm -f -- "${RUNS:?}" "${REDS:?}" "${LEDGER:?}"' EXIT
page=1 runs_ok=1
while :; do
    body=$(get_etag "repos/$REPO/actions/workflows/ci.yml/runs?event=merge_group&created=$START..$LAST&per_page=100&page=$page") \
        || { runs_ok=0; break; }
    n=$(jq '.workflow_runs | length' <<< "$body")
    jq -c '.workflow_runs[] | {id, head_branch, head_sha, status, conclusion, run_started_at, updated_at,
             day: (.created_at[0:10])}' <<< "$body" >> "$RUNS"
    [ "$n" -lt 100 ] && break
    page=$((page + 1))
done

# ---- 2. tag every red -----------------------------------------------------------------------------
# cause rows: run_id day pr cause wall_kill source
tag_run() {
    local id=$1 day=$2 pr=$3 jobs gate xjobs ann log res
    jobs=$(get_fixed "repos/$REPO/actions/runs/$id/jobs?per_page=100") || { printf '%s\t%s\t%s\tnot_measured\t0\tnone\n' "$id" "$day" "$pr"; return; }
    local ids j
    # wall kill and section/step causes from the annotations of every failed or cancelled job
    ids=$(jq -r '.jobs[] | select(.conclusion == "failure" or .conclusion == "cancelled") | "\(.id)\t\(.name)\t\(.conclusion)"' <<< "$jobs")
    ann=""
    while IFS=$'\t' read -r j name concl; do
        [ -n "$j" ] || continue
        a=$(get_fixed "repos/$REPO/check-runs/$j/annotations?per_page=100") || a='[]'
        ann+=$(jq -r --arg job "$name" '.[] | select(.annotation_level == "failure")
                 | "\($job)\t\(.title // "")\t\(.message | gsub("[\t\n]"; " "))"' <<< "$a")$'\n'
    done <<< "$ids"
    # section results: the results.json output the gate job read (X86:/DET: lines of its env block)
    gate=$(jq -r '.jobs[] | select(.name == "gate") | select(.conclusion != "skipped") | .id' <<< "$jobs" | head -1)
    res=""
    if [ -n "$gate" ] && log=$(get_fixed "repos/$REPO/actions/jobs/$gate/logs"); then
        res=$(printf '%s\n' "$log" | sed -nE 's/^[^ ]+ +(X86|DET): (\{.*\})\r?$/\1\t\2/p' \
            | while IFS=$'\t' read -r tag json; do
                job=x86-main; [ "$tag" = DET ] && job=determinism
                jq -r --arg job "$job" 'to_entries[] | select(.value.continue_on_error | not)
                    | select(.value.result == "failure" or .value.result == "cancelled")
                    | "\($job)/\(.key)"' <<< "$json" 2> /dev/null
              done)
    fi
    # wall kill, per job: the fat_driver spawn line or the runner's maximum-execution-time cancel
    local wkjobs
    wkjobs=$(awk -F'\t' '$3 ~ /step exceeded its timeout/ {print $1 "\tstep"}
                  $3 ~ /exceeded the maximum execution time/ {print $1 "\tjob"}' <<< "$ann" | sort -u)
    local src=results
    if [ -z "$res" ]; then
        src=annotations
        res=$(awk -F'\t' 'match($3, /^section [^ :]+: (failure|cancelled)$/) {
                  s = $3; sub(/^section /, "", s); sub(/:.*/, "", s); print (s == "a" ? $1 : $1 "/" s) }' <<< "$ann")
    fi
    if [ -z "$res" ]; then
        # a sectioned job that failed with no section result: the runner's own failure annotation names it
        # (lost communication, shutdown signal, ...); "Process completed with exit code" says nothing
        src=annotations
        res=$(awk -F'\t' '($1 == "x86-main" || $1 == "determinism") && $3 !~ /^Process completed with exit code/ {
                  m = $3; sub(/\. .*/, "", m); print $1 "/" substr(m, 1, 60); exit }' <<< "$ann")
    fi
    # guard step detail: "::error title=<manifest job>: guard step fail::<step>  [reason]"
    local steps
    steps=$(awk -F'\t' '$2 ~ /: guard step (fail|timeout)$/ {
                j = $2; sub(/: guard step.*/, "", j); s = $3; sub(/  \[.*$/, "", s); k = ($2 ~ /timeout$/) ? "timeout" : "fail"; print j "\t" s "\t" k }' <<< "$ann")
    # jobs outside the sectioned ones: a failure, or a cancel that was a wall kill. workspace-test only
    # aggregates its shards, so it is never a cause of its own.
    local plain
    plain=$(jq -r --arg wk "$(cut -f1 <<< "$wkjobs")" '.jobs[]
                 | select(.conclusion == "failure" or (.conclusion == "cancelled" and (.name as $n | $wk | split("\n") | index($n))))
                 | .name | select(. != "x86-main" and . != "determinism" and . != "gate" and . != "ci / gate"
                                  and . != "workspace-test")' <<< "$jobs")
    local any=0 c st w
    while read -r c; do
        [ -n "$c" ] || continue
        any=1
        # the first failing guard step of that section (manifest job = <section>-steps or <section>)
        st=$(awk -F'\t' -v s="${c#*/}" '$1 == s || $1 == s "-steps" {print $2; exit}' <<< "$steps")
        w=$(awk -F'\t' -v j="${c%%/*}" '$1 == j {k = k (k ? "+" : "") $2} END {print k ? k : 0}' <<< "$wkjobs")
        # a section that failed on a guard step that itself reported FAIL was not the one the deadline killed
        if [ -n "$st" ] && [ "$(awk -F'\t' -v s="${c#*/}" '($1 == s || $1 == s "-steps") {print $3; exit}' <<< "$steps")" = fail ]; then w=0; fi
        printf '%s\t%s\t%s\t%s%s\t%s\t%s\n' "$id" "$day" "$pr" "$c" "${st:+/$st}" "$w" "$src"
    done <<< "$res"
    while read -r c; do
        [ -n "$c" ] || continue
        any=1; w=$(awk -F'\t' -v j="$c" '$1 == j {k = k (k ? "+" : "") $2} END {print k ? k : 0}' <<< "$wkjobs")
        printf '%s\t%s\t%s\t%s\t%s\tjobs\n' "$id" "$day" "$pr" "$c" "$w"
    done <<< "$plain"
    [ "$any" = 1 ] || printf '%s\t%s\t%s\tuntagged\t0\tnone\n' "$id" "$day" "$pr"
}

# host-hours of a run: the sum over its jobs that ran of completed - started
run_hours() {
    get_fixed "repos/$REPO/actions/runs/$1/jobs?per_page=100" 2> /dev/null | jq '[.jobs[]
        | select(.conclusion != "skipped" and .started_at != null and .completed_at != null)
        | ((.completed_at | fromdateiso8601) - (.started_at | fromdateiso8601)) | select(. > 0)] | add // 0 | . / 3600'
}

# good code: the PR merged, and the change it merged is byte-for-byte the change the failed queue commit
# tested (same `git patch-id --stable` of commit-vs-parent; the queue squashes, so one parent each side).
# Reads git objects only; fetching by sha writes no ref and no FETCH_HEAD.
good_code() { # <pr> <failed queue commit sha>
    local p m a b
    p=$(get_fixed "repos/$REPO/pulls/$1") || { echo unknown; return; }
    m=$(jq -r 'if .merged_at then .merge_commit_sha else "" end' <<< "$p")
    [ -n "$m" ] || { echo no; return; }
    git cat-file -e "$2^{commit}" 2> /dev/null && git cat-file -e "$m^{commit}" 2> /dev/null \
        || git fetch -q --no-write-fetch-head origin "$2" "$m" 2> /dev/null || { echo unknown; return; }
    a=$(git diff "$2^" "$2" 2> /dev/null | git patch-id --stable | cut -d' ' -f1)
    b=$(git diff "$m^" "$m" 2> /dev/null | git patch-id --stable | cut -d' ' -f1)
    if [ -z "$a" ] || [ -z "$b" ]; then echo unknown; elif [ "$a" = "$b" ]; then echo yes; else echo no; fi
}

LOST=0 LOST_RUNS=0 LOST_UNKNOWN=0
while read -r r; do
    concl=$(jq -r '.conclusion // ""' <<< "$r")
    [ "$concl" = failure ] || [ "$concl" = cancelled ] || continue
    id=$(jq -r .id <<< "$r"); day=$(jq -r .day <<< "$r"); hs=$(jq -r .head_sha <<< "$r")
    pr=$(jq -r '.head_branch' <<< "$r" | sed -nE 's#^gh-readonly-queue/[^/]+/pr-([0-9]+)-.*#\1#p')
    tag_run "$id" "$day" "${pr:-?}" | sort -u >> "$REDS"
    if [ "$concl" = failure ] && [ -n "$pr" ]; then
        case "$(good_code "$pr" "$hs")" in
            yes) h=$(run_hours "$id") && LOST=$(awk -v a="$LOST" -v b="$h" 'BEGIN{print a+b}') && LOST_RUNS=$((LOST_RUNS + 1)) ;;
            unknown) LOST_UNKNOWN=$((LOST_UNKNOWN + 1)) ;;
        esac
    fi
done < "$RUNS"

# ---- 3. repeats against the lesson file ------------------------------------------------------------
REPEATS=0 REPEAT_LINES="" LESSON_ROWS=0
if [ -f "$LESSONS" ]; then
    LESSON_ROWS=$(grep -cvE '^(#|$)' "$LESSONS")
    while IFS=$'\t' read -r rid rday _ cause rwk _; do
        wkf=""; [ "$rwk" = 0 ] || wkf=" wallkill:$rwk"
        while IFS=$'\t' read -r glob ref since; do
            case "$glob" in '#'* | '') continue ;; esac
            # shellcheck disable=SC2053  # glob match on purpose
            if [[ $cause$wkf == $glob ]] && [[ $rday > $since ]]; then
                REPEATS=$((REPEATS + 1)); REPEAT_LINES+="  $rid $cause -> $ref"$'\n'; break
            fi
        done < "$LESSONS"
    done < "$REDS"
fi

# ---- 4. merges per day ----------------------------------------------------------------------------
MERGES="" MIN_M="" SUM_M=0 merges_ok=1 d=$START
while [[ $d < $END ]]; do
    if m=$(get_etag "search/issues?q=repo:$REPO+is:pr+is:merged+merged:$d&per_page=1" | jq -r .total_count) && [[ $m =~ ^[0-9]+$ ]]; then
        MERGES+="$d=$m "; SUM_M=$((SUM_M + m))
        if [ -z "$MIN_M" ] || [ "$m" -lt "$MIN_M" ]; then MIN_M=$m; fi
    else
        merges_ok=0; MERGES+="$d=not_measured "
    fi
    d=$(date -u -d "$d + 1 day" +%F)  # bashrs disable-line=DET002 (calendar arithmetic, no clock read)
done

# ---- 5. print ------------------------------------------------------------------------------------
CALLS=$(grep -c "^call$" "$LEDGER") HITS=$(grep -c "^hit$" "$LEDGER") READ_FAIL=$(grep -c "^fail$" "$LEDGER")
cnt() { jq -s --arg c "$1" '[.[] | select(.status == "completed" and .conclusion == $c)] | length' "$RUNS"; }
TOTAL=$(jq -s '[.[] | select(.status == "completed")] | length' "$RUNS")
OPEN=$(jq -s '[.[] | select(.status != "completed")] | length' "$RUNS")
SUC=$(cnt success) FAIL=$(cnt failure) CAN=$(cnt cancelled)
DAYS_SEEN=$(jq -rs '[.[].day] | unique | length' "$RUNS")
WK=$(awk -F'\t' '$5 != "0" {print $1}' "$REDS" | sort -u | wc -l)
UNTAG=$(awk -F'\t' '$4 == "untagged" || $4 == "not_measured"' "$REDS" | wc -l)
if [ $((SUC + FAIL)) -gt 0 ]; then
    PASS=$(awk -v s="$SUC" -v f="$FAIL" 'BEGIN{printf "%.1f", 100*s/(s+f)}')
else PASS=not_measured; fi

[ -n "$TSV" ] && { printf 'run_id\tday\tpr\tcause\twall_kill\tsource\n'; cat "$REDS"; } > "$TSV"

nm() { [ "$1" = 1 ] && printf '%s' "$2" || printf 'not_measured'; }
echo "ci morning report  $START..$LAST ($DAYS UTC days)  repo $REPO"
echo "  github: $CALLS calls ($HITS free: cache or 304), $READ_FAIL failed reads"
printf '  queue runs        %s  (success %s, failure %s, cancelled %s; %s still running, not counted)\n' \
    "$(nm "$runs_ok" "$TOTAL")" "$SUC" "$FAIL" "$CAN" "$OPEN"
printf '  pass rate         %s%%  (success / (success + failure))\n' "$PASS"
printf '  wall kills        %s runs\n' "$WK"
printf '  repeats           %s  (lesson file %s: %s rows)\n' "$REPEATS" "${LESSONS##*/}" "$LESSON_ROWS"
[ -n "$REPEAT_LINES" ] && printf '%s' "$REPEAT_LINES"
printf '  host-hours lost   %.1f h on %s runs that failed on good code (%s not decidable)\n' "$LOST" "$LOST_RUNS" "$LOST_UNKNOWN"
printf '  merges/day        min %s, mean %s  [%s]\n' "${MIN_M:-not_measured}" \
    "$(awk -v s="$SUM_M" -v n="$DAYS" 'BEGIN{printf "%.1f", s/n}')" "${MERGES% }"
echo "  reds by cause (machine-tagged; $UNTAG untagged):"
awk -F'\t' '!seen[$1 FS $4 FS $5]++ {k=$4 ($5 != "0" ? "  [wall kill: " $5 "]" : ""); n[k]++; ids[k]=ids[k] " " $1} END {for (k in n) printf "    %3d  %s  runs%s\n", n[k], k, ids[k]}' "$REDS" | sort -rn

ok() { if [ "$1" = 1 ]; then printf 'ok'; else printf 'FAIL'; fi; }
b1=0; [ "$DAYS_SEEN" -ge 7 ] && [ "$DAYS" -ge 7 ] && b1=1
b2=0; [ "$runs_ok" = 1 ] && [ "$TOTAL" -ge 50 ] && b2=1
b3=0; [ "$merges_ok" = 1 ] && [ -n "$MIN_M" ] && [ "$MIN_M" -ge 15 ] && b3=1
b4=0; [ "$PASS" != not_measured ] && awk -v p="$PASS" 'BEGIN{exit !(p >= 90)}' && b4=1
b5=0; [ "$WK" -eq 0 ] && [ "$UNTAG" -eq 0 ] && b5=1
b6=0; [ "$REPEATS" -eq 0 ] && [ "$LESSON_ROWS" -gt 0 ] && [ "$UNTAG" -eq 0 ] && b6=1
[ "$READ_FAIL" -eq 0 ] || { b2=0; b5=0; b6=0; }
m=NO; [ $((b1 & b2 & b3 & b4 & b5 & b6)) = 1 ] && m=YES
printf '  mastered: %s  days %s/7 %s; runs %s/50 %s; merges min %s/15 %s; pass %s%%/90 %s; wall kills %s/0 %s; repeats %s/0 %s\n' \
    "$m" "$DAYS_SEEN" "$(ok $b1)" "$TOTAL" "$(ok $b2)" "${MIN_M:-?}" "$(ok $b3)" "$PASS" "$(ok $b4)" "$WK" "$(ok $b5)" "$REPEATS" "$(ok $b6)"
[ "$READ_FAIL" -eq 0 ] && [ "$runs_ok" = 1 ] && [ "$merges_ok" = 1 ]
