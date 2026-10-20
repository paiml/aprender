#!/usr/bin/env bash
# check_queue_merge_plain.sh -- every script that ARMS a pull request on a merge-queue repo calls plain
# `gh pr merge <pr>`, with no method flag and no --auto. #5794, from the #5789 loss of 2026-10-20.
#
# THE DEFECT. #5789's auto-merge was armed at 12:39Z with method MERGE on a queue repo whose queue method is
# SQUASH. GitHub accepted the request and the PR never queued; a plain `gh pr merge` queued it at once. A method
# flag on a queue repo is not a preference, it is a second source of truth that can disagree with the queue.
#
# Usage:  bash scripts/check_queue_merge_plain.sh            scan scripts/, exit 1 on a flagged line
#         bash scripts/check_queue_merge_plain.sh --self-test  the case table, including the RED row
set -euo pipefail

# Lines that name a method flag on purpose: fixtures that stub gh and classify its calls, not arming paths.
ALLOW='scripts/check_release_host_receipts.sh scripts/release/rehearse.sh'

# flagged_lines FILE...  -> "file:line:text" for each non-comment line that runs `pr merge` with a method flag
flagged_lines() {
    local f
    for f in "$@"; do
        grep -nE 'pr merge' "$f" 2> /dev/null \
            | grep -vE '^[0-9]+:[[:space:]]*#' \
            | grep -E -- '(^|[[:space:]])--(squash|merge|rebase|auto)([[:space:]]|$)' \
            | sed "s|^|$f:|" || true
    done
}

scan() {
    local root="$1" f out="" skip
    while IFS= read -r f; do
        skip=0
        for a in $ALLOW; do [ "$f" = "$a" ] && skip=1; done
        [ "$f" = scripts/check_queue_merge_plain.sh ] && skip=1
        [ "$skip" = 1 ] && continue
        out+="$(cd "$root" && flagged_lines "$f")"$'\n'
    done < <(cd "$root" && find scripts -name '*.sh' -type f | sort)
    printf '%s' "$out" | sed '/^$/d'
}

self_test() {
    local td fails=0 got
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    mkdir -p "$td/scripts"
    printf 'gh pr merge "$pr" --repo "$repo"\n# gh pr merge --squash --auto is the old way\n' > "$td/scripts/ok.sh"
    got="$(scan "$td")"
    if [ -z "$got" ]; then echo "ok    plain merge and a comment are clean"; else echo "FAIL  plain merge flagged: $got"; fails=1; fi
    for flag in --squash --merge --rebase --auto; do
        printf 'gh pr merge "$pr" %s\n' "$flag" > "$td/scripts/bad.sh"
        got="$(scan "$td")"
        if [ -n "$got" ]; then echo "ok    RED on method flag $flag"; else echo "FAIL  $flag was NOT flagged"; fails=1; fi
    done
    printf 'out=$(gh pr merge "$N" --repo r --squash --auto 2>&1)\n' > "$td/scripts/bad.sh"
    got="$(scan "$td")"
    if [ -n "$got" ]; then echo "ok    RED on a flag mid-line (the arm_pr_automerge shape)"; else echo "FAIL  mid-line flag not flagged"; fails=1; fi
    got="$(scan "$(git rev-parse --show-toplevel)")"
    if [ -z "$got" ]; then echo "ok    the real tree has no method flag on an arming path"; else echo "FAIL  real tree: $got"; fails=1; fi
    return "$fails"
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    "")
        root="$(git rev-parse --show-toplevel)"
        bad="$(scan "$root")"
        if [ -n "$bad" ]; then
            printf 'FAIL  a script arms with a method flag on a merge-queue repo (#5794):\n%s\n' "$bad"; exit 1
        fi
        echo "ok    no arming script passes a method flag to gh pr merge" ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
