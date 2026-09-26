#!/usr/bin/env bash
# check_nightly_liveness.sh - NIGHTLY-RED that counts CANCELLED runs (OBS-02, #4489).
#
# WHY THIS EXISTS
# ---------------
# qwen-story-daily's last 14+ scheduled runs ended "cancelled", and cuda-nightly's
# yoga job was cancelled too, so no scheduled model run happened - and nothing
# said so. NIGHTLY-RED (APR-RELEASE-001 section 614) was a report line with no
# implementation, and every reader of it counted FAILURES. A run the job
# timeout or a concurrency group cancels never fails: it has no verdict at all,
# which is exactly the state an alert must not read as quiet.
#
# The mechanisms (five-whys in contracts/apr-nightly-liveness-v1.yaml):
#   qwen-story-daily  pmat reached the gx10 runner, which switched on the
#                     per-beat bug-hunt; one `pmat query --churn` measured 407 s
#                     and the story runs 3 queries x ~20 paths, so the 30-min job
#                     timeout cancelled the run inside Beat 3. Fixed by bounding
#                     the hunt (scripts/lib_story_pmat.sh).
#   cuda-nightly      the ada-yoga job waits on the cross-workflow `perf-yoga`
#                     concurrency group, and a group holds ONE pending job: a
#                     newer arrival cancels the pending one. cancel-in-progress:
#                     false protects only the RUNNING job.
#
# RULE: a model nightly whose newest N (default 2) completed SCHEDULED runs are
# all non-success - cancelled, failure, timed_out, anything but success - is RED.
# So is one with no scheduled run in the last LIVENESS_STALE_H hours: a nightly
# that stopped firing is as dead as one that fails. In-progress runs are
# ignored, and a manual (workflow_dispatch) rerun does not make the SCHEDULE
# alive.
#
# USAGE
#   bash scripts/check_nightly_liveness.sh            # offline case table (CI)
#   bash scripts/check_nightly_liveness.sh --live [workflow.yml ...]
#       queries GitHub; prints one NIGHTLY-RED / nightly-ok line per workflow;
#       exit 1 if any is RED. Default workflows: the model nightlies below.
#   bash scripts/check_nightly_liveness.sh --runs-json FILE WORKFLOW [--now EPOCH]
#       judge one saved `actions/workflows/<wf>/runs` response.
#
# The bare run is the offline case table because guard_tree.sh runs every
# check_*.sh on every PR: a PR gate must not go red because last night's
# nightly did. It includes a planted mutant - the failure-only count - which
# must turn the table RED.

set -uo pipefail

REPO="${LIVENESS_REPO:-paiml/aprender}"
RED_AT="${LIVENESS_RED_AT:-2}"
STALE_H="${LIVENESS_STALE_H:-50}"
MODEL_NIGHTLIES=(cuda-nightly.yml qwen-story-daily.yml)

# What counts toward the streak. The mutant in the case table rewrites this line.
COUNTED='. != "success"'

# The jq program is single-quoted so the shell never touches it; only the
# counted predicate is spliced in, from the one line the mutant rewrites.
JUDGE_JQ='
def streak: reduce .[] as $c ({n: 0, open: true};
    if .open and ($c | counted) then .n += 1 else .open = false end) | .n;
[(.workflow_runs // [])[] | select(.event == "schedule" and .status == "completed")]
| sort_by(.created_at) | reverse
| (map(.conclusion // "none")) as $c
| ($c | streak) as $n
| if length == 0 then
    "RED NIGHTLY-RED \($wf) no completed scheduled run on record"
  elif ($now - (.[0].created_at | fromdateiso8601)) > $stale_s then
    "RED NIGHTLY-RED \($wf) stale: newest scheduled run \(.[0].created_at) is older than \($stale)h"
  elif $n >= $red then
    "RED NIGHTLY-RED \($wf) \($n) consecutive non-success (\($c[0:$n] | group_by(.) | map("\(.[0])=\(length)") | join(" "))) newest=\(.[0].created_at) run=\(.[0].id // "?")"
  else
    "OK nightly-ok \($wf) streak=\($n) newest=\($c[0]) \(.[0].created_at)"
  end'

# judge FILE WORKFLOW NOW_EPOCH -> prints one verdict line; returns 1 when RED.
judge() {
    local file="$1" wf="$2" now="$3" out
    out=$(jq -r --arg wf "$wf" --argjson now "$now" --argjson red "$RED_AT" \
        --argjson stale "$STALE_H" --argjson stale_s "$((STALE_H * 3600))" "def counted: $COUNTED; $JUDGE_JQ" "$file") \
        || { printf 'NIGHTLY-RED %s unreadable runs json %s\n' "$wf" "$file"; return 1; }
    printf '%s\n' "${out#* }"
    case "$out" in RED*) return 1 ;; esac
    return 0
}

# fixture FILE NOW "conclusion ..." [event] [status-of-newest]
# Newest first, one day apart, all scheduled+completed unless overridden.
fixture() {
    local file="$1" now="$2" list="$3" ev="${4:-schedule}" st="${5:-completed}"
    local i=0 c json="" sep="" ts e s
    for c in $list; do
        ts=$(date -u -d "@$((now - 3600 - i * 86400))" +%Y-%m-%dT%H:%M:%SZ)
        e="schedule"; s="completed"
        if [ "$i" -eq 0 ]; then e="$ev"; s="$st"; fi
        json="$json$sep{\"id\":$((100 + i)),\"event\":\"$e\",\"status\":\"$s\",\"conclusion\":\"$c\",\"created_at\":\"$ts\"}"
        sep=","; i=$((i + 1))
    done
    printf '{"workflow_runs":[%s]}\n' "$json" > "$file"
}

case_table() {
    local tmp now fails=0 row list want ev st got
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/nightly-liveness.XXXXXX") || return 1
    now=$(date -u +%s)  # bashrs disable-line=DET002 (fixture clock, never an artifact)
    # want | newest-first conclusions | newest event | newest status
    while IFS='|' read -r want list ev st; do
        [ -n "$want" ] || continue
        fixture "$tmp/runs.json" "$now" "$list" "$ev" "$st"
        if judge "$tmp/runs.json" fx.yml "$now" >/dev/null; then got=OK; else got=RED; fi
        if [ "$got" = "$want" ]; then
            printf '  ok    %-3s <- [%s] %s %s\n' "$want" "$list" "$ev" "$st"
        else
            printf '  FAIL  want %s got %s <- [%s] %s %s\n' "$want" "$got" "$list" "$ev" "$st"
            fails=$((fails + 1))
        fi
    done <<'ROWS'
OK|success|schedule|completed
OK|cancelled success|schedule|completed
RED|cancelled cancelled success|schedule|completed
RED|cancelled cancelled cancelled cancelled|schedule|completed
RED|failure failure|schedule|completed
RED|cancelled failure success|schedule|completed
RED|timed_out cancelled|schedule|completed
OK|success cancelled cancelled|schedule|completed
RED|success cancelled cancelled|workflow_dispatch|completed
RED|success cancelled cancelled|schedule|in_progress
OK|success success|schedule|in_progress
RED|skipped neutral success|schedule|completed
RED|action_required null|schedule|completed
ROWS
    # No scheduled run at all, and a schedule that stopped firing.
    printf '{"workflow_runs":[]}\n' > "$tmp/runs.json"
    if judge "$tmp/runs.json" fx.yml "$now" >/dev/null; then
        printf '  FAIL  want RED got OK <- no runs\n'; fails=$((fails + 1))
    else printf '  ok    RED <- no runs\n'; fi
    # Stale boundary: the newest run is 1h before the fixture clock, so judging
    # at +48h makes it 49h old (live) and at +50h 51h old (stale, > 50h).
    fixture "$tmp/runs.json" "$now" "success success"
    if judge "$tmp/runs.json" fx.yml "$((now + 48 * 3600))" >/dev/null; then
        printf '  ok    OK  <- newest success 49h old\n'
    else printf '  FAIL  want OK got RED <- newest success 49h old\n'; fails=$((fails + 1)); fi
    if judge "$tmp/runs.json" fx.yml "$((now + 50 * 3600))" >/dev/null; then
        printf '  FAIL  want RED got OK <- newest success 51h old\n'; fails=$((fails + 1))
    else printf '  ok    RED <- newest success 51h old (stale)\n'; fi
    rm -rf "${tmp:?}" || return 1
    return "$fails"
}

self_prove() {
    local fails=0 tmp mut
    echo "check_nightly_liveness: case table"
    case_table || fails=$?
    # The planted falsifier of apr-nightly-liveness-v1: the pre-#4489 reading of
    # NIGHTLY-RED counted failures only. That counter must fail this table (it
    # calls two cancelled nights GREEN); if it passes, the table cannot tell a
    # cancelled nightly from a live one.
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/nightly-liveness-mut.XXXXXX") || return 1
    mut="$tmp/mutant.sh"
    sed "s/^COUNTED='. != \"success\"'\$/COUNTED='. == \"failure\"'/" "$0" > "$mut"
    if cmp -s "$0" "$mut"; then
        echo "  FAIL  the failure-only mutant did not apply (COUNTED line moved?)"
        fails=$((fails + 1))
    elif LIVENESS_MUTANT=1 bash "$mut" >/dev/null 2>&1; then
        echo "  FAIL  the failure-only mutant PASSED the table - cancelled nights read as GREEN"
        fails=$((fails + 1))
    else
        echo "  ok    the failure-only mutant (old NIGHTLY-RED) turns the table RED"
    fi
    rm -rf "${tmp:?}" || return 1
    echo
    if [ "$fails" -eq 0 ]; then
        echo "check_nightly_liveness: OK - cancelled, failed, stale and missing nightlies are RED"
        return 0
    fi
    echo "check_nightly_liveness: $fails row(s) FAILED"
    return 1
}

live() {
    local wf tmp red=0 now
    [ "$#" -gt 0 ] || set -- "${MODEL_NIGHTLIES[@]}"
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/nightly-liveness-live.XXXXXX") || return 1
    now=$(date -u +%s)  # bashrs disable-line=DET002 (the live verdict's clock)
    for wf in "$@"; do
        if ! gh api "repos/$REPO/actions/workflows/$wf/runs?event=schedule&per_page=15" \
                > "$tmp/runs.json" 2> "$tmp/err"; then
            # Fail closed: an unreadable nightly is not a live one.
            printf 'NIGHTLY-RED %s could not list runs: %s\n' "$wf" "$(head -c 200 "$tmp/err")"
            red=1; continue
        fi
        judge "$tmp/runs.json" "$wf" "$now" || red=1
    done
    rm -rf "${tmp:?}" || return 1
    return "$red"
}

case "${1:-}" in
    "")          if [ -n "${LIVENESS_MUTANT:-}" ]; then case_table; else self_prove; fi ;;
    --live)      shift; live "$@" ;;
    --runs-json) [ "$#" -ge 3 ] || { echo "usage: --runs-json FILE WORKFLOW [--now EPOCH]" >&2; exit 2; }
                 n=$(date -u +%s)  # bashrs disable-line=DET002
                 [ "${4:-}" != "--now" ] || n="$5"
                 judge "$2" "$3" "$n" ;;
    -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed '$d' ;;
    *)           echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac
