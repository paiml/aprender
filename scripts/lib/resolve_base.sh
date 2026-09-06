# resolve_base.sh - name the base a change is judged against, on every CI checkout shape.
#
# Source it (option-neutral: no `set` here; failure is the return status):
#     REPO_ROOT=<toplevel> PROG=<caller> . scripts/lib/resolve_base.sh || exit 1
#     resolve_base HEAD && printf '%s (%s)\n' "$BASE_REF" "$BASE_HOW"
#
# Extracted verbatim from scripts/check_roadmap_diff_additive.sh (G-6, PMAT-980,
# #2874), whose 15-row self-test is its case table, so the shipped-path ratchet
# (check_hardcoded_paths.sh, PMAT-1059) and any later differential guard judge
# HEAD against the SAME base: merge-base(origin/main, HEAD) when history is
# there; the merge commit's first parent on a depth-1 pull_request checkout; the
# single parent that IS the origin/main tip on a merge_group squash head; and a
# refusal - never the branch against itself - for anything else.
# ROADMAP_DIFF_FORCE_SHALLOW=1 makes a full clone behave like depth-1 (tests).
resolve_base() {
    local head=$1 mb parents
    mb=""
    if [ "${ROADMAP_DIFF_FORCE_SHALLOW:-0}" != 1 ]; then
        mb=$(git -C "$REPO_ROOT" merge-base origin/main "$head" 2>/dev/null || true)
    fi
    if [ -n "$mb" ]; then BASE_REF="$mb"; BASE_HOW="merge-base(origin/main, $head)"; return 0; fi
    # read the parents off the commit OBJECT: in a depth-1 clone `rev-list --parents` shows none (shallow graft), `cat-file -p` still does
    parents=$(git -C "$REPO_ROOT" cat-file -p "$head^{commit}" 2>/dev/null | awk '/^parent /{printf "%s ", $2} /^$/{exit}')
    local p1 main_tip; p1=$(printf '%s\n' "$parents" | cut -d' ' -f1); main_tip=$(git -C "$REPO_ROOT" rev-parse origin/main)
    if [ "$(printf '%s\n' "$parents" | wc -w)" -lt 2 ]; then
        # merge_group: the queue's temporary head is a SINGLE-parent squash-shaped commit whose parent is
        # the base branch tip (run 34002682350: parent == origin/main). That parent is the base. Any other
        # single-parent head (a branch commit) has no nameable base here and is refused.
        if [ -n "$p1" ] && [ "$p1" = "$main_tip" ]; then
            BASE_REF="$p1"; BASE_HOW="single parent == origin/main tip (merge_group squash head, shallow checkout)"; return 0
        fi
        printf '%s: merge-base(origin/main, %s) is unresolvable (shallow checkout) and %s is not a merge commit nor a commit on the origin/main tip,\n' "$PROG" "$head" "$head" >&2
        printf '    so no base can be named. A pull_request job checks out refs/pull/N/merge, a merge_group job the queue head; run with an explicit <base> otherwise.\n' >&2
        return 1
    fi
    if git -C "$REPO_ROOT" cat-file -e "$p1^{commit}" 2>/dev/null; then
        BASE_REF="$p1"; BASE_HOW="first parent of the merge commit $head (shallow checkout)"
    else
        BASE_REF=$(git -C "$REPO_ROOT" rev-parse origin/main); BASE_HOW="origin/main tip (shallow checkout; the merge commit's first parent is not fetched)"
    fi
    return 0
}
