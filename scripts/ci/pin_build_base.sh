#!/usr/bin/env bash
# pin_build_base.sh - pin refs/remotes/origin/main to the commit THIS run was built on,
# the comparand of every ratchet in guard-tree and guard-cargo (#4861).
#
# THE DEFECT. The guard sections fetched origin/main's tip when they STARTED, and a
# section can start hours after the merge commit it judges was built. #4813's run built
# 092b5e990 (a merge into 358e9d4d11); its guard-tree started when main was ece9e563bd,
# three merges later. One of those (da7059c5c7) shrank scripts/pipe_grep_q_baseline.txt
# from 62 to 60. The ratchet compared the run's tree (62, the base's own value; #4813
# never touched the file) with the tip (60) and reported "GREW". Nothing in the change
# grew: main moved under it.
#
# THE RULE. A ratchet judges the change against the base the run was built on:
#   pull_request  the first parent of the merge commit the job checked out (what GitHub
#                 actually built; a differing pull_request.base.sha is printed, never used
#                 in its place). A pull_request HEAD that is not a two-parent merge is
#                 REFUSED: there is no base to name, and the tip is the defect above.
#   merge_group   merge_group.base_sha, as the refinement gate already pins it (#4502).
#   anything else (push, tag, workflow_dispatch): unchanged - the origin/main tip, and on
#                 push deepened by one so scripts/lib/resolve_base.sh can take the first
#                 parent.
#
# NOT A RELAXATION. The tip was "stricter" in one way: it also refused re-adding an entry
# main had deleted since the base. That case is still refused, where it can be judged
# correctly: the merge_group run is built on main as it is when it merges, and pins THAT
# base. A pull_request run cannot know main's future; judging it by main's present was a
# false red on every branch older than the latest shrink.
#
# Usage: pin_build_base.sh              (CI; GITHUB_EVENT_NAME, PR_BASE_SHA, MG_BASE_SHA)
#        pin_build_base.sh --self-test  (case table; temp repos only, no network)
# Exit:  0 pinned, 1 refused (no nameable base) or self-test failed, 2 usage.
set -euo pipefail

ZERO=0000000000000000000000000000000000000000

is_sha() { [[ "${1:-}" =~ ^[0-9a-f]{40}$ ]] && [ "$1" != "$ZERO" ]; }

# The parents of HEAD, read off the commit OBJECT: at depth 1 the shallow graft hides them
# from rev-list; cat-file still shows them.
head_parents() {
    git cat-file -p 'HEAD^{commit}' | awk '/^parent /{print $2} /^$/{exit}'
}

pin() { # pin <sha> <how>
    git fetch -q --no-tags --depth=1 origin "$1"
    git update-ref refs/remotes/origin/main "$1"
    printf 'pin_build_base: %s: origin/main pinned to %s (%s)\n' "${GITHUB_EVENT_NAME:-?}" \
        "$(git rev-parse 'refs/remotes/origin/main^{commit}')" "$2"
}

run() {
    local ev="${GITHUB_EVENT_NAME:-}" parents n p1
    case "$ev" in
        pull_request)
            parents=$(head_parents)
            n=$(printf '%s\n' "$parents" | grep -c . || true)
            if [ "$n" != 2 ]; then
                printf '::error::pin_build_base: pull_request HEAD %s has %s parent(s), not the two of the merge commit GitHub builds; no build base can be named, and the origin/main tip is not one (#4861)\n' \
                    "$(git rev-parse --short HEAD)" "$n" >&2
                return 1
            fi
            p1=$(printf '%s\n' "$parents" | sed -n 1p)
            if is_sha "${PR_BASE_SHA:-}" && [ "$PR_BASE_SHA" != "$p1" ]; then
                printf 'pin_build_base: note: pull_request.base.sha %s is not the merge commit'"'"'s first parent; the first parent is what was built\n' "$PR_BASE_SHA"
            fi
            pin "$p1" "first parent of the merge commit $(git rev-parse --short HEAD)"
            ;;
        merge_group)
            if ! is_sha "${MG_BASE_SHA:-}"; then
                printf '::error::pin_build_base: merge_group event carries no base_sha (got "%s")\n' "${MG_BASE_SHA:-}" >&2
                return 1
            fi
            pin "$MG_BASE_SHA" "merge_group.base_sha"
            ;;
        *)
            git fetch -q --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main
            # push shape: HEAD IS the origin/main tip, so every differential ratchet judges it against
            # its FIRST PARENT (scripts/lib/resolve_base.sh: HEAD vs HEAD would pass vacuously, the
            # G-10 quorum's finding), and a depth-1 checkout has not fetched that parent. Deepen by one.
            if [ "$ev" = push ]; then git fetch -q --no-tags --deepen=1 origin +refs/heads/main:refs/remotes/origin/main; fi
            printf 'pin_build_base: %s: origin/main is the tip %s (no event base for this event)\n' "${ev:-?}" \
                "$(git rev-parse 'refs/remotes/origin/main^{commit}')"
            ;;
    esac
}

self_test() {
    local here lib T fails=0 rows=0 A TIP
    here=$(cd "$(dirname "$0")" && pwd)
    lib="$here/../lib_baseline_ratchet.sh"
    [ -f "$lib" ] || { printf 'self-test: %s missing\n' "$lib" >&2; return 2; }
    T=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf '${T:?}'" EXIT
    local B=scripts/pipe_grep_q_baseline.txt O="$T/origin"
    export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
    export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
    unset GITHUB_EVENT_NAME PR_BASE_SHA MG_BASE_SHA BASELINE_RATCHET_BASE_REF

    # origin: main A (62) -> B (60, the shrink) -> C (unrelated). PR 1 forks at A and does not
    # touch the baseline (#4813's shape); PR 2 forks at A and really grows it to 63. GitHub built
    # each merge commit when main was A.
    git init -q -b main "$O"
    mkdir -p "$O/scripts"
    printf '# count\n62\n' > "$O/$B"; git -C "$O" add -A; git -C "$O" commit -qm A
    A=$(git -C "$O" rev-parse HEAD)
    git -C "$O" checkout -q -b pr "$A"
    printf 'x\n' > "$O/feature.txt"; git -C "$O" add -A; git -C "$O" commit -qm H
    git -C "$O" checkout -q -b grow "$A"
    printf '# count\n63\n' > "$O/$B"; git -C "$O" add -A; git -C "$O" commit -qm G
    git -C "$O" checkout -q --detach "$A"
    git -C "$O" merge -q --no-ff -m M1 pr; git -C "$O" update-ref refs/pull/1/merge HEAD
    git -C "$O" checkout -q --detach "$A"
    git -C "$O" merge -q --no-ff -m M2 grow; git -C "$O" update-ref refs/pull/2/merge HEAD
    git -C "$O" checkout -q main
    printf '# count\n60\n' > "$O/$B"; git -C "$O" add -A; git -C "$O" commit -qm B
    printf 'y\n' > "$O/other.txt"; git -C "$O" add -A; git -C "$O" commit -qm C
    git -C "$O" config uploadpack.allowAnySHA1InWant true
    TIP=$(git -C "$O" rev-parse main)

    clone() { # clone <dir> <ref>: a depth-1 checkout of <ref>, as actions/checkout makes it
        git init -q "$1"; git -C "$1" remote add origin "file://$O"
        git -C "$1" fetch -q --no-tags --depth=1 origin "$2"; git -C "$1" checkout -q --detach FETCH_HEAD
    }
    row() { # row <name> <want> <got>
        rows=$((rows + 1))
        if [ "$2" = "$3" ]; then printf '  ok    %s\n' "$1"
        else printf '  FAIL  %s: want %s, got %s\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
    }
    ratchet() { # rc of the REAL baseline ratchet in <dir> against its origin/main
        if ( cd "$1" && . "$lib" && baseline_ratchet_check "$1" "$B" count ) >/dev/null 2>&1; then echo 0; else echo 1; fi
    }
    pinrc() { # pinrc <dir> [VAR=val...]: rc of this script's CI mode in <dir>
        local d=$1; shift
        if ( cd "$d" && env "$@" bash "$here/pin_build_base.sh" ) >/dev/null 2>&1; then echo 0; else echo 1; fi
    }

    # 1. The defect, reproduced: #4813's shape under the OLD step (fetch the tip) reds.
    clone "$T/old" refs/pull/1/merge
    git -C "$T/old" fetch -q --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main
    row 'old step: #4813 shape (base 62, main since shrank to 60) reds GREW - the defect' 1 "$(ratchet "$T/old")"
    # 2. The cure: the same checkout, pinned, is green, and the pin names the build base.
    clone "$T/pr" refs/pull/1/merge
    row 'pull_request: pin exits 0' 0 "$(pinrc "$T/pr" GITHUB_EVENT_NAME=pull_request PR_BASE_SHA="$A")"
    row 'pull_request: origin/main == the merge commit first parent' "$A" "$(git -C "$T/pr" rev-parse origin/main)"
    row 'pull_request: #4813 shape is green against its build base' 0 "$(ratchet "$T/pr")"
    # 3. Not a relaxation: a branch that really grows the baseline still reds when pinned.
    clone "$T/grow" refs/pull/2/merge
    pinrc "$T/grow" GITHUB_EVENT_NAME=pull_request >/dev/null
    row 'pull_request: real growth (62 -> 63) still reds against the build base' 1 "$(ratchet "$T/grow")"
    # 4. A base.sha in the payload that differs from what was built never replaces it.
    clone "$T/stale" refs/pull/1/merge
    pinrc "$T/stale" GITHUB_EVENT_NAME=pull_request PR_BASE_SHA="$TIP" >/dev/null
    row 'pull_request: base.sha != first parent -> the first parent wins' "$A" "$(git -C "$T/stale" rev-parse origin/main)"
    # 5. A pull_request HEAD that is not a merge commit is refused, never pinned to the tip.
    clone "$T/nomerge" pr
    row 'pull_request: non-merge HEAD refused' 1 "$(pinrc "$T/nomerge" GITHUB_EVENT_NAME=pull_request)"
    row 'pull_request: a refused run leaves no origin/main' none "$(git -C "$T/nomerge" rev-parse -q --verify origin/main || echo none)"
    # 6. merge_group pins base_sha; a missing or all-zero base_sha is refused.
    clone "$T/mg" refs/pull/1/merge
    row 'merge_group: pin exits 0' 0 "$(pinrc "$T/mg" GITHUB_EVENT_NAME=merge_group MG_BASE_SHA="$TIP")"
    row 'merge_group: origin/main == base_sha' "$TIP" "$(git -C "$T/mg" rev-parse origin/main)"
    row 'merge_group: a base_sha main has shrunk past reds the stale entry (the queue still catches it)' 1 "$(ratchet "$T/mg")"
    row 'merge_group: all-zero base_sha refused' 1 "$(pinrc "$T/mg" GITHUB_EVENT_NAME=merge_group MG_BASE_SHA=$ZERO)"
    row 'merge_group: empty base_sha refused' 1 "$(pinrc "$T/mg" GITHUB_EVENT_NAME=merge_group MG_BASE_SHA=)"
    # 7. Other events keep the tip; push also deepens by one (its first parent is fetched).
    clone "$T/push" main
    row 'push: pin exits 0' 0 "$(pinrc "$T/push" GITHUB_EVENT_NAME=push)"
    row 'push: origin/main is the tip' "$TIP" "$(git -C "$T/push" rev-parse origin/main)"
    row 'push: first parent fetched (deepened by one)' 0 "$(git -C "$T/push" cat-file -e "$TIP^1^{commit}" 2>/dev/null && echo 0 || echo 1)"
    clone "$T/wd" main
    pinrc "$T/wd" GITHUB_EVENT_NAME=workflow_dispatch >/dev/null
    row 'workflow_dispatch: origin/main is the tip' "$TIP" "$(git -C "$T/wd" rev-parse origin/main)"

    printf 'pin_build_base self-test: %d row(s), %d failed\n' "$rows" "$fails"
    [ "$rows" -gt 0 ] && [ "$fails" = 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    '') run ;;
    *) printf 'usage: %s [--self-test]\n' "$0" >&2; exit 2 ;;
esac
