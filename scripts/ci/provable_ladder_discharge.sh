#!/usr/bin/env bash
# provable_ladder_discharge.sh — the provable-ladder `discharge` and `summary-fresh` steps
# (ci/sections.yml), split by event per operator ruling C312 (#4578, 2026-10-04):
#
#   pull_request, merge_group  -> mode pr:   build.sh, then `pv discharge check --strict
#                                 --comparator`. Lake elaboration and the comparator run;
#                                 the leanchecker re-check (~75 min under load) does NOT.
#   anything else (push to main, workflow_dispatch, schedule, unknown)
#                              -> mode full: `pv discharge run` — build, check --strict,
#                                 comparator AND leanchecker, and it rewrites the summary.
#
# An unknown or empty event is `full`: a mistake here can only add the slow check, never
# drop it. Planted rows: scripts/ci/ladder_pr_skips_leanchecker.sh.
#
#   provable_ladder_discharge.sh mode           -> prints pr|full for $EVENT
#   provable_ladder_discharge.sh discharge      -> runs the mode's discharge; exit is pv's
#   provable_ladder_discharge.sh summary-fresh  -> full: the committed summary is what the run
#                                                  wrote; pr: it was written for this Lean tree
#
# env: EVENT, PV, LEAN, SUMMARY, LOCK (flock file), LOG (discharge log path)
# EXIT 0 accept · 1 reject · 2 decline (pv's and build.sh's convention) · 64 usage
set -uo pipefail

mode() {
    case "${EVENT:-}" in
        pull_request | merge_group) echo pr ;;
        *) echo full ;;
    esac
}

discharge() {
    local m rc
    : "${PV:?}" "${LEAN:?}" "${LOCK:?}" "${LOG:?}"
    m="$(mode)"
    echo "provable-ladder mode: $m (event: ${EVENT:-<unset>})"
    if [ "$m" = full ]; then
        flock "$LOCK" "$PV" discharge run "$LEAN" > "$LOG" 2>&1
        rc=$?
    else
        # One lock for both: build.sh unpacks into the shared .lake that check reads.
        flock "$LOCK" bash -c '
            cd "$1" && bash build.sh || exit $?
            cd - > /dev/null && exec "$2" discharge check "$1" --strict --comparator
        ' _ "$LEAN" "$PV" > "$LOG" 2>&1
        rc=$?
    fi
    tail -n 40 "$LOG"
    return "$rc"
}

# The committed summary's tree_sha, read without jq (not on every runner).
committed_tree_sha() {
    git show "HEAD:$SUMMARY" 2>/dev/null |
        sed -n 's/^ *"tree_sha": *"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1
}

summary_fresh() {
    local want got
    : "${LEAN:?}" "${SUMMARY:?}"
    if [ "$(mode)" = full ]; then
        git diff --exit-code -- "$SUMMARY"
        return $?
    fi
    # pr mode wrote no summary (`check` never does), so a `git diff` would pass vacuously.
    # What a PR CAN be held to: the committed summary was generated for the Lean tree it
    # ships. Its leanchecker verdict is re-derived on the push to main.
    want="$(git rev-parse "HEAD:$LEAN" 2>/dev/null)"
    got="$(committed_tree_sha)"
    if [ -n "$want" ] && [ "$want" = "$got" ]; then
        echo "ok    $SUMMARY tree_sha = HEAD:$LEAN ($want); leanchecker runs on the push to main"
        return 0
    fi
    echo "FAIL  $SUMMARY tree_sha '${got:-<none>}' != HEAD:$LEAN '${want:-<none>}': regenerate it with pv discharge run"
    return 1
}

case "${1:-}" in
    mode) mode ;;
    discharge) discharge ;;
    summary-fresh) summary_fresh ;;
    *) echo "usage: $0 mode|discharge|summary-fresh" >&2; exit 64 ;;
esac
