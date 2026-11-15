#!/usr/bin/env bash
# Case table for scripts/merge_freeze.sh (#6494). Pure jq fixtures, no network.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tool="$here/merge_freeze.sh"
tmp=$(mktemp -d)
trap 'rm -rf "${tmp:?}"' EXIT
NOW=2026-11-15T12:00:00Z
fails=0

# case NAME NIGHTLY_JSON QUEUE_JSON EXTRA_ARGS WANT_RC WANT_FIRST_LINE_PREFIX [WANT_HOLDS]
case_() {
    local name=$1 n=$2 q=$3 extra=$4 rc_want=$5 verdict_want=$6 holds_want=${7:-}
    printf '%s' "$n" >"$tmp/n.json"
    printf '%s' "$q" >"$tmp/q.json"
    local out rc
    # shellcheck disable=SC2086
    out=$(bash "$tool" --nightly "$tmp/n.json" --queue "$tmp/q.json" --now "$NOW" $extra 2>&1) && rc=0 || rc=$?
    local first holds
    first=$(printf '%s\n' "$out" | head -n 1)
    holds=$(printf '%s\n' "$out" | grep -o '^HOLD pr=[0-9]*' | sed 's/HOLD pr=//' | tr '\n' ',' || true)
    if [ "$rc" != "$rc_want" ] || [ "${first%% *}" != "$verdict_want" ] || [ "$holds" != "$holds_want" ]; then
        echo "FAIL $name: rc=$rc want=$rc_want first='$first' holds='$holds' want='$holds_want'" >&2
        fails=$((fails + 1))
    else
        echo "ok   $name"
    fi
}

G='[{"status":"completed","conclusion":"success","createdAt":"2026-11-15T03:00:00Z"}]'
R='[{"status":"completed","conclusion":"failure","createdAt":"2026-11-15T03:00:00Z"}]'
Q='[{"number":10,"labels":["P1"]},{"number":11,"labels":["P0"]},{"number":12,"labels":[]}]'

case_ green-opens "$G" "$Q" "" 0 OPEN
case_ red-freezes-and-holds "$R" "$Q" "" 3 FREEZE "10,12,"
case_ red-exempt-flag "$R" "$Q" "--exempt 12" 3 FREEZE "10,"
case_ red-empty-queue "$R" '[]' "" 3 FREEZE
case_ timed-out-is-red '[{"status":"completed","conclusion":"timed_out","createdAt":"2026-11-15T03:00:00Z"}]' "$Q" "" 3 FREEZE "10,12,"
case_ cancelled-skipped-then-green '[{"status":"completed","conclusion":"cancelled","createdAt":"2026-11-15T09:00:00Z"},{"status":"completed","conclusion":"success","createdAt":"2026-11-15T03:00:00Z"}]' "$Q" "" 0 OPEN
case_ unordered-input-newest-wins '[{"status":"completed","conclusion":"success","createdAt":"2026-11-14T03:00:00Z"},{"status":"completed","conclusion":"failure","createdAt":"2026-11-15T03:00:00Z"}]' "$Q" "" 3 FREEZE "10,12,"
case_ in-progress-ignored '[{"status":"in_progress","conclusion":null,"createdAt":"2026-11-15T09:00:00Z"},{"status":"completed","conclusion":"failure","createdAt":"2026-11-15T03:00:00Z"}]' "$Q" "" 3 FREEZE "10,12,"
case_ no-runs-not-measured '[]' "$Q" "" 5 NOT_MEASURED
case_ stale-verdict-not-measured '[{"status":"completed","conclusion":"success","createdAt":"2026-11-13T03:00:00Z"}]' "$Q" "" 5 NOT_MEASURED
case_ malformed-not-measured 'not json' "$Q" "" 5 NOT_MEASURED
case_ malformed-queue-not-measured "$R" 'oops' "" 5 NOT_MEASURED

# missing file -> NOT_MEASURED, never OPEN
out=$(bash "$tool" --nightly "$tmp/absent.json" --queue "$tmp/q.json" --now "$NOW" 2>&1) && rc=0 || rc=$?
if [ "$rc" = 5 ] && [ "${out%% *}" = NOT_MEASURED ]; then echo "ok   missing-file"; else echo "FAIL missing-file rc=$rc" >&2; fails=$((fails + 1)); fi

[ "$fails" = 0 ] || { echo "merge_freeze: $fails case(s) failed" >&2; exit 1; }
echo "merge_freeze: all cases passed"
