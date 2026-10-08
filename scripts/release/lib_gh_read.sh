#!/usr/bin/env bash
# lib_gh_read.sh -- a GitHub READ that one or two failed calls do not end (#4939).
#
# autopilot.sh read GitHub with bare `gh` calls. One failed read either stopped the pass
# (the merge-commit read: "#PR has no merge commit") or quietly became an empty answer,
# and the next pass re-ran lanes that had already been measured. gh_read tries a read up
# to GH_READ_TRIES times (3), GH_READ_WAIT_S seconds apart (10), each try killed after
# GH_READ_TIMEOUT_S seconds (25). It prints the answer of the first try that exits 0 and
# returns the last try's status when none does.
#
# The worst case is bounded, not hoped for: TRIES * (TIMEOUT + 5 s kill grace) + (TRIES - 1)
# * WAIT = 3*30 + 2*10 = 110 s with the defaults, and a read that fails fast twice and then
# answers costs 2 * WAIT = 20 s plus its three calls. gh_read_worst_s prints the bound for the
# settings in force; the planted case P1 in scripts/check_release_gh_read.sh measures the
# fail-twice cost against it.
#
# READS ONLY. A write (release create/edit, workflow run, issue comment/close, pr create,
# api -X ...) is never retried: a write that timed out may have landed, and repeating it
# double-dispatches. gh_read refuses those forms (status 2) instead of running them.
#
# SOURCED, so OPTION-NEUTRAL: no `set` here (check_sourced_libs_option_neutral.sh).
#     . "$REPO_ROOT/scripts/release/lib_gh_read.sh" || exit 2
#     MC=$(gh_read pr view "$PR" --repo "$REPO" --json mergeCommit -q .mergeCommit.oid)

# gh_read_is_write ARGS...: status 0 when ARGS is a gh form that changes something.
gh_read_is_write() {
    case "${1:-} ${2:-}" in
        "release create"|"release edit"|"release delete"|"release upload"|"workflow run"|\
        "issue comment"|"issue close"|"issue create"|"issue edit"|"pr create"|"pr merge"|\
        "pr edit"|"pr comment"|"pr close"|"run rerun"|"run cancel") return 0 ;;
    esac
    [ "${1:-}" = api ] || return 1
    local a
    for a in "$@"; do
        case "$a" in -X|--method|-f|-F|--field|--raw-field|--input|-XPOST|-XPATCH|-XPUT|-XDELETE) return 0 ;; esac
    done
    return 1
}

# gh_read_worst_s: the longest one gh_read can take with the settings in force, in seconds.
gh_read_worst_s() {
    local n=${GH_READ_TRIES:-3} w=${GH_READ_WAIT_S:-10} t=${GH_READ_TIMEOUT_S:-25}
    printf '%s\n' $(( n * (t + 5) + (n - 1) * w ))
}

# gh_read ARGS...: run `gh ARGS...`, retrying a failed read; stdout of the try that answered.
gh_read() {
    if gh_read_is_write "$@"; then
        printf 'gh_read: refusing to retry a write: gh %s\n' "$*" >&2
        return 2
    fi
    local n=${GH_READ_TRIES:-3} w=${GH_READ_WAIT_S:-10} t=${GH_READ_TIMEOUT_S:-25} i=1 out rc
    while :; do
        out=$(timeout -k 5 "$t" gh "$@") && { printf '%s\n' "$out"; return 0; }
        rc=$?  # read here, never after an `if`: an if with no else leaves $? at 0
        [ "$i" -ge "$n" ] && return "$rc"
        printf 'gh_read: try %s of %s failed (rc %s): gh %s\n' "$i" "$n" "$rc" "$*" >&2
        i=$((i + 1))
        sleep "$w"
    done
}
