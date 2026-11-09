#!/usr/bin/env bash
# free_own_target_dir.sh - remove this job's own cargo target dir at job end (#5397).
#
# A CI job builds into `targets/aprender-ci/<PR_OR_REF>/run-<RUN_ID>` (#1693).
# Nothing freed that dir when the job ended: a later run of the same PR reclaimed
# it, and a PR with no later run kept it. 27 PR trees held 263 GB on one host,
# and a host at 90% disk drops its clean-room runners (C338).
#
# WHAT IT REMOVES. Exactly `<root>/<PR_OR_REF>/run-<RUN_ID>`. Never the PR parent
# (sibling runs of the PR may be live there: that is why #1693 made the dir
# per-run) and never another run's dir.
#
# WHERE IT RUNS. An `if: always()` last step of the job, so it fires on success,
# failure and cancel. The step comes after the job's sections and teardown, so
# no container of this job still mounts the dir.
#
# A FAILED REMOVAL IS A WARNING, NOT A RED: the dir is only reclaimable space.
# The warning is an annotation on the job, so it is visible; the host sweep and
# the next run's reclaim still collect what this step could not.
#
#   free_own_target_dir.sh <pr_or_ref> <run_id>
#   free_own_target_dir.sh --self-test
#
# EXIT  0 freed, nothing to free, or removal failed (warning printed)
#       1 bad arguments: an unsafe name or a non-numeric run id

set -uo pipefail

ROOT=${APR_CI_TARGET_ROOT:-/mnt/nvme-raid0/targets/aprender-ci}
IMAGE=${IMAGE:-localhost:5000/sovereign-ci:stable}

free_own() {
    local pr_or_ref=$1 run_id=$2 dir parent
    if [ -z "$pr_or_ref" ] || ! [[ $run_id =~ ^[0-9]+$ ]]; then
        echo "free_own_target_dir: need <pr_or_ref> and a numeric <run_id>" >&2
        return 1
    fi
    # PR_OR_REF is a PR number or a ref name ("gh-readonly-queue/main/pr-N-sha"
    # has slashes), so slashes are fine; anything that climbs or re-roots is not.
    case "/$pr_or_ref/" in
        *"/../"* | *"/./"* | *"//"*)
            echo "free_own_target_dir: refusing unsafe name '$pr_or_ref'" >&2
            return 1
            ;;
    esac
    case "$pr_or_ref" in /* | -*)
        echo "free_own_target_dir: refusing unsafe name '$pr_or_ref'" >&2
        return 1
        ;;
    esac
    parent="$ROOT/$pr_or_ref"
    dir="$parent/run-$run_id"
    if [ -L "$dir" ]; then
        echo "free_own_target_dir: refusing symlink $dir" >&2
        return 1
    fi
    if [ ! -d "$dir" ]; then
        echo "free_own_target_dir: nothing to free ($dir)"
        return 0
    fi
    case "$dir" in "$ROOT"/?*/run-[0-9]*) ;; *)
        echo "free_own_target_dir: refusing path $dir" >&2
        return 1
        ;;
    esac
    rm -rf -- "${dir:?}" 2>/dev/null
    # Files a container wrote as root survive a plain rm: retry as root, in the
    # job image, with only the PR parent mounted and only run-<id> named.
    if [ -d "$dir" ] && command -v docker >/dev/null 2>&1; then
        docker run --rm -v "$parent:/p" "$IMAGE" rm -rf -- "/p/run-$run_id" 2>/dev/null
    fi
    if [ -d "$dir" ]; then
        echo "::warning::free_own_target_dir: could not remove $dir (host sweep will)"
        return 0
    fi
    echo "free_own_target_dir: freed $dir"
}

self_test() {
    local t fails=0
    t=$(mktemp -d) || return 2
    ROOT=$t/aprender-ci
    chk() { # <name> <condition-rc>
        if [ "$2" -eq 0 ]; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
    }
    mkdir -p "$ROOT/5390-s1/run-11" "$ROOT/5390-s1/run-12" "$ROOT/gh-readonly-queue/main/pr-1-abc/run-21"
    : >"$ROOT/5390-s1/run-11/f"
    free_own 5390-s1 11 >/dev/null
    chk "own dir removed" "$([ ! -e "$ROOT/5390-s1/run-11" ]; echo $?)"
    chk "sibling run kept" "$([ -d "$ROOT/5390-s1/run-12" ]; echo $?)"
    chk "PR parent kept" "$([ -d "$ROOT/5390-s1" ]; echo $?)"
    free_own gh-readonly-queue/main/pr-1-abc 21 >/dev/null
    chk "slashed ref name removed" "$([ ! -e "$ROOT/gh-readonly-queue/main/pr-1-abc/run-21" ]; echo $?)"
    free_own 5390-s1 99 >/dev/null
    chk "absent dir is rc 0" "$?"
    free_own ../x 1 >/dev/null 2>&1
    chk "dot-dot refused" "$([ $? -eq 1 ]; echo $?)"
    free_own /etc 1 >/dev/null 2>&1
    chk "absolute refused" "$([ $? -eq 1 ]; echo $?)"
    free_own "" 1 >/dev/null 2>&1
    chk "empty name refused" "$([ $? -eq 1 ]; echo $?)"
    free_own 5390-s1 "1;x" >/dev/null 2>&1
    chk "non-numeric run id refused" "$([ $? -eq 1 ]; echo $?)"
    ln -s "$t" "$ROOT/5390-s1/run-77"
    free_own 5390-s1 77 >/dev/null 2>&1
    chk "symlink refused and target kept" "$([ $? -eq 1 ] && [ -d "$t" ]; echo $?)"
    rm -rf -- "${t:?}"
    [ "$fails" -eq 0 ] && echo "self-test: all ok" || echo "self-test: $fails FAILED"
    return $((fails > 0))
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi
[ $# -eq 2 ] || { echo "usage: ${0##*/} <pr_or_ref> <run_id> | --self-test" >&2; exit 1; }
free_own "$1" "$2"
