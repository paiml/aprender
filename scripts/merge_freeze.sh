#!/usr/bin/env bash
# merge_freeze.sh -- freeze/drain verdict for the merge queue (#6494). Operator ruling 2026-11-15: a red nightly
# freezes merges until it is green. READ-ONLY: it prints a verdict and the entries to hold; it never calls gh or
# edits the queue. The caller feeds it two JSON files (e.g. `gh run list --json status,conclusion,createdAt` and the
# queue as [{number, labels:[...]}]) and acts on the exit code.
#
# Verdict, from the NEWEST completed nightly run (cancelled/skipped runs are not a verdict; in-progress is ignored):
#   success            -> OPEN          exit 0
#   failure/timed_out  -> FREEZE        exit 3, then one `HOLD pr=N` per queue entry that is not exempt
#   anything else      -> NOT_MEASURED  exit 5 (no completed run, run older than 36h, unreadable input): never OPEN
# Exempt (never held): label P0, or a number named by --exempt N[,N...].
# Usage: merge_freeze.sh --nightly FILE --queue FILE [--now ISO8601Z] [--exempt N,N]
set -euo pipefail

nightly="" queue="" now="" exempt=""
while [ $# -gt 0 ]; do
    case "$1" in
        --nightly) nightly=${2:?}; shift 2 ;;
        --queue) queue=${2:?}; shift 2 ;;
        --now) now=${2:?}; shift 2 ;;
        --exempt) exempt=${2:?}; shift 2 ;;
        *) echo "NOT_MEASURED reason=unknown-arg:$1"; exit 5 ;;
    esac
done
[ -n "$now" ] || now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
nm() { echo "NOT_MEASURED reason=$1"; exit 5; }

[ -r "$nightly" ] || nm "nightly-unreadable"
[ -r "$queue" ] || nm "queue-unreadable"
jq -e 'type == "array"' "$nightly" >/dev/null 2>&1 || nm "nightly-malformed"
jq -e 'type == "array"' "$queue" >/dev/null 2>&1 || nm "queue-malformed"

verdict=$(jq -r --arg now "$now" '
  [ .[] | select(.status == "completed" and (.conclusion == "success" or .conclusion == "failure" or .conclusion == "timed_out")) ]
  | sort_by(.createdAt) | last
  | if . == null then "NONE"
    elif (($now | fromdateiso8601) - (.createdAt | fromdateiso8601)) > 129600 then "STALE"
    elif .conclusion == "success" then "GREEN" else "RED" end' "$nightly") || nm "nightly-malformed"

case "$verdict" in
    GREEN) echo "OPEN reason=latest-nightly-green"; exit 0 ;;
    NONE) nm "no-completed-nightly" ;;
    STALE) nm "latest-nightly-older-than-36h" ;;
    RED) ;;
    *) nm "nightly-malformed" ;;
esac

echo "FREEZE reason=latest-nightly-red"
jq -r --arg ex "$exempt" '
  ($ex | split(",") | map(select(. != "") | tonumber)) as $e
  | .[] | select((.labels | index("P0")) == null and (.number as $n | $e | index($n)) == null)
  | "HOLD pr=\(.number)"' "$queue" || nm "queue-malformed"
exit 3
