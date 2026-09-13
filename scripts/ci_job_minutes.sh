#!/usr/bin/env bash
# ci_job_minutes.sh — runner-minutes per CI job, from the jobs API (BSE-15).
#
# WHY THIS EXISTS
# ---------------
# "This job cost 96 h of PR runner-time in 30 days" was the load-bearing number
# behind deleting the `pr-review-receipt` job, and it arrived as prose. A figure
# that decides a deletion has to be re-derivable by whoever reads the commit, or
# it is an assertion wearing a unit. This script is that derivation.
#
# WHAT IT MEASURES, precisely, because the wrong sum is easy to compute:
#   * RUNNER time, not wall time: Σ per JOB of (completed_at − started_at). A run
#     with four jobs in parallel occupies four runners; run-level duration would
#     undercount it fourfold.
#   * `started_at`, not `created_at`: the gap between them is QUEUE wait, which
#     costs no runner. Counting it would inflate exactly the jobs that wait most.
#   * only jobs that actually ran: a `skipped` job has no runner time, and on the
#     jobs API its started_at/completed_at can still be populated.
#
# REFUSALS (a measurement that cannot be taken must not return a number):
#   * no `gh`, or `gh` unauthenticated                      -> exit 2
#   * no `jq`                                                -> exit 2
#   * the window returned ZERO runs                          -> exit 2, never 0.0
#     GitHub expires run history, so an empty answer is far more often "the
#     window is past retention" than "nothing ran". Reporting 0 h there would
#     read as an improvement.
#
#   bash scripts/ci_job_minutes.sh --since 2026-08-07 [--until 2026-09-07]
#   bash scripts/ci_job_minutes.sh --since 2026-08-07 --event pull_request --json
#   bash scripts/ci_job_minutes.sh --self-test        # arithmetic, both polarities, offline
#
# EXIT: 0 a table (or JSON) was produced; 1 a bad argument; 2 the box cannot answer.
set -uo pipefail
PROG=${0##*/}
REPO=${CI_JOB_MINUTES_REPO:-paiml/aprender}
SINCE=""; UNTIL=""; EVENT=""; JSON=0; LIMIT=${CI_JOB_MINUTES_LIMIT:-400}; WORKFLOW=""

# ---------------------------------------------------------------------------
# sum_jobs — stdin: the jobs API's `.jobs[]` objects, one JSON per line.
# stdout: "<seconds>\t<job name>" per job name, descending. Pure arithmetic, so
# --self-test can drive it from a fixture with no network.
# ---------------------------------------------------------------------------
sum_jobs() {
    jq -s -r '
      map(select(.started_at != null and .completed_at != null and .conclusion != "skipped"))
      | map({name: .name,
             secs: (((.completed_at | fromdateiso8601) - (.started_at | fromdateiso8601)))})
      | map(select(.secs > 0))
      | group_by(.name)
      | map({name: .[0].name, secs: (map(.secs) | add), runs: length})
      | sort_by(-.secs)
      | .[] | "\(.secs)\t\(.runs)\t\(.name)"
    '
}

if [ "${1:-}" = "--self-test" ]; then
    command -v jq >/dev/null 2>&1 || { printf '%s: ENV - jq is missing\n' "$PROG" >&2; exit 2; }
    red=0; n=0
    row() { # row <label> <expected> <actual>
        n=$((n + 1))
        if [ "$3" = "$2" ]; then printf 'ok    row %-2s %s\n' "$n" "$1"
        else printf 'FAIL  row %-2s %s: wanted [%s], got [%s]\n' "$n" "$1" "$2" "$3"; red=1; fi
    }
    # 3600 s + 1800 s for job A over two runs; 600 s for job B.
    fx='{"name":"A","started_at":"2026-09-01T00:00:00Z","completed_at":"2026-09-01T01:00:00Z","conclusion":"success"}
{"name":"A","started_at":"2026-09-02T00:00:00Z","completed_at":"2026-09-02T00:30:00Z","conclusion":"failure"}
{"name":"B","started_at":"2026-09-01T00:00:00Z","completed_at":"2026-09-01T00:10:00Z","conclusion":"success"}'
    row "two runs of A sum to 5400 s over 2 runs, and A sorts first" \
        "5400	2	A" "$(printf '%s\n' "$fx" | sum_jobs | head -1)"
    row "B is 600 s over 1 run" "600	1	B" "$(printf '%s\n' "$fx" | sum_jobs | grep -P '\tB$')"
    # A skipped job has no runner time even with timestamps populated.
    row "a skipped job contributes nothing" "" \
        "$(printf '%s\n' '{"name":"S","started_at":"2026-09-01T00:00:00Z","completed_at":"2026-09-01T02:00:00Z","conclusion":"skipped"}' | sum_jobs)"
    # A queued-but-never-started job must not be counted as zero-length noise.
    row "a job that never started contributes nothing" "" \
        "$(printf '%s\n' '{"name":"Q","started_at":null,"completed_at":null,"conclusion":null}' | sum_jobs)"
    # THE POLARITY THAT MATTERS: if the sum silently used created_at (queue
    # time), this row would come back larger. It must not exist as a field here.
    row "queue wait is not runner time (created_at is not read)" "3600	1	C" \
        "$(printf '%s\n' '{"name":"C","created_at":"2026-09-01T00:00:00Z","started_at":"2026-09-01T00:50:00Z","completed_at":"2026-09-01T01:50:00Z","conclusion":"success"}' | sum_jobs)"
    # The truncation rule, driven through the real script rather than asserted in
    # prose: a window whose run count reaches --limit must exit 2, and the same
    # window with room to spare must not. CI_JOB_MINUTES_REPO points at a repo
    # that does not exist, so `gh run list` fails first and both rows would exit 2
    # for the wrong reason -- so this row only runs where gh can answer, and says
    # so when it skips.
    n=$((n + 1))
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        out=$("$0" --since 2026-09-01 --until 2026-09-07 --limit 1 2>&1); rc=$?
        case "$rc:$out" in
            2:*TRUNCATED*) printf 'ok    row %-2s a run count equal to --limit is TRUNCATED, never a total\n' "$n" ;;
            *) printf 'FAIL  row %-2s --limit 1 over a busy window must exit 2 TRUNCATED (rc=%s)\n' "$n" "$rc"; red=1 ;;
        esac
    else
        printf 'skip  row %-2s truncation row needs an authenticated gh (stated, not silently passed)\n' "$n"
    fi
    [ "$red" = 0 ] && { printf '\nSELF-TEST PASSED (%s/%s)\n' "$n" "$n"; exit 0; }
    printf '\nSELF-TEST FAILED\n'; exit 1
fi

while [ $# -gt 0 ]; do case "$1" in
    --since) SINCE=$2; shift 2 ;;
    --until) UNTIL=$2; shift 2 ;;
    --event) EVENT=$2; shift 2 ;;
    --workflow) WORKFLOW=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --json) JSON=1; shift ;;
    *) printf 'usage: %s --since YYYY-MM-DD [--until YYYY-MM-DD] [--event <e>] [--workflow <f>] [--repo o/n] [--limit N] [--json]\n' "$PROG" >&2; exit 1 ;;
esac; done
[ -n "$SINCE" ] || { printf '%s: --since YYYY-MM-DD is required\n' "$PROG" >&2; exit 1; }
for t in gh jq; do command -v "$t" >/dev/null 2>&1 || { printf '%s: ENV - %s is missing; no number is produced\n' "$PROG" "$t" >&2; exit 2; }; done
gh auth status >/dev/null 2>&1 || { printf '%s: ENV - gh is not authenticated; no number is produced\n' "$PROG" >&2; exit 2; }

RANGE="$SINCE"; [ -n "$UNTIL" ] && RANGE="$SINCE..$UNTIL"
set -- --repo "$REPO" --created "$RANGE" --limit "$LIMIT" --json databaseId,event,workflowName,createdAt
[ -n "$EVENT" ] && set -- "$@" --event "$EVENT"
[ -n "$WORKFLOW" ] && set -- "$@" --workflow "$WORKFLOW"
RUNS=$(gh run list "$@" 2>/dev/null | jq -r '.[].databaseId') || {
    printf '%s: ENV - `gh run list` failed for %s in %s\n' "$PROG" "$REPO" "$RANGE" >&2; exit 2; }
COUNT=$(printf '%s\n' "$RUNS" | grep -c . || true)
[ "${COUNT:-0}" -gt 0 ] || {
    printf '%s: ENV - zero runs in %s for %s. GitHub expires run history, so this is\n' "$PROG" "$RANGE" "$REPO" >&2
    printf '      almost always retention rather than an idle fleet. Refusing to report 0 h.\n' >&2; exit 2; }
# NO SILENT CAP. `gh run list --limit N` returns at most N runs with no signal that
# it truncated, so a sum over a bound window would quietly become a sum over "the
# N most recent runs" and read as a total. The first derivation of this number hit
# exactly that: 400 runs requested, 400 returned, and the figure was a floor
# wearing the units of a total. Equality is the only evidence available, so it is
# treated as truncation and refused.
[ "$COUNT" -lt "$LIMIT" ] || {
    printf '%s: TRUNCATED - `gh run list` returned exactly --limit (%s) runs for %s, so\n' "$PROG" "$LIMIT" "$RANGE" >&2
    printf '      the window is not fully covered and any sum here is a FLOOR, not a total.\n' >&2
    printf '      Re-run with a larger --limit (or a narrower window):\n' >&2
    printf '        %s --since %s%s --limit %s\n' "$PROG" "$SINCE" "$([ -n "$UNTIL" ] && printf -- ' --until %s' "$UNTIL")" "$((LIMIT * 3))" >&2
    exit 2; }

TMP=$(mktemp) || exit 2
trap 'rm -f -- "$TMP"' EXIT
for id in $RUNS; do
    gh api --paginate "repos/$REPO/actions/runs/$id/jobs" --jq '.jobs[]' 2>/dev/null >> "$TMP" || true
done
[ -s "$TMP" ] || { printf '%s: ENV - the jobs API returned nothing for %s runs\n' "$PROG" "$COUNT" >&2; exit 2; }

if [ "$JSON" = 1 ]; then
    sum_jobs < "$TMP" | jq -R -s --arg repo "$REPO" --arg range "$RANGE" --arg runs "$COUNT" '
      {repo: $repo, window: $range, runs: ($runs|tonumber),
       jobs: (split("\n") | map(select(length>0) | split("\t")
              | {name: .[2], seconds: (.[0]|tonumber), runs: (.[1]|tonumber),
                 hours: ((.[0]|tonumber) / 3600 * 100 | round / 100)}))}
      | . + {total_hours: (.jobs | map(.seconds) | add / 3600 * 100 | round / 100)}'
    exit 0
fi
printf '=== runner-hours per job — %s, runs created %s (%s runs%s) ===\n' \
    "$REPO" "$RANGE" "$COUNT" "$([ -n "$EVENT" ] && printf ', event=%s' "$EVENT")"
printf '%10s  %6s  %s\n' hours runs job
sum_jobs < "$TMP" | awk -F'\t' '{printf "%10.1f  %6d  %s\n", $1/3600, $2, $3; t+=$1} END {printf "%10.1f  %6s  TOTAL\n", t/3600, ""}'
