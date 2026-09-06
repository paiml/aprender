# resolve_base.sh - name the base a change is judged against, on every CI checkout shape.
#
# Source it (option-neutral: no `set` here; failure is the return status):
#     REPO_ROOT=<toplevel>; PROG=<caller>; . scripts/lib/resolve_base.sh || exit 1
#     (plain assignments: `PROG=x . file` does NOT persist past the `.` builtin, and
#     an unset PROG under `set -u` would kill the CALLER inside resolve_base — R7 of
#     check_hardcoded_paths.sh passed for exactly that wrong reason on 2026-09-06)
#     resolve_base HEAD && printf '%s (%s)\n' "$BASE_REF" "$BASE_HOW"
#
# Extracted verbatim from scripts/check_roadmap_diff_additive.sh (G-6, PMAT-980,
# #2874), whose 15-row self-test is its case table, so the shipped-path ratchet
# (check_hardcoded_paths.sh, PMAT-1059) and any later differential guard judge
# HEAD against the SAME base: merge-base(origin/main, HEAD) when history is
# there; the merge commit's first parent on a depth-1 pull_request checkout; the
# single parent that IS the origin/main tip on a merge_group squash head; the
# FIRST PARENT when HEAD itself is the origin/main tip (a push to main: the
# merge-base would be HEAD, and HEAD judged against HEAD is a vacuous pass —
# the G-10 review quorum's blocking finding, 2026-09-06); and a refusal -
# never the tree against itself - for anything else.
# ROADMAP_DIFF_FORCE_SHALLOW=1 makes a full clone behave like depth-1 (tests).
resolve_base() {
    local head=$1 mb parents headid main_tip
    local PROG="${PROG:-resolve_base}"
    headid=$(git -C "$REPO_ROOT" rev-parse "$head^{commit}" 2>/dev/null || true)
    main_tip=$(git -C "$REPO_ROOT" rev-parse origin/main 2>/dev/null || true)
    if [ -n "$headid" ] && [ "$headid" = "$main_tip" ]; then
        # push shape: HEAD IS the origin/main tip. merge-base(origin/main, HEAD) is HEAD, and a
        # differential of HEAD against itself passes vacuously. The change under judgment is
        # what the tip's first parent lacks; a depth-1 checkout must deepen by one to hold it.
        local p; p=$(git -C "$REPO_ROOT" rev-parse -q --verify "$head^1^{commit}" 2>/dev/null || true)
        if [ -n "$p" ]; then BASE_REF="$p"; BASE_HOW="first parent of $head (HEAD is the origin/main tip: push shape)"; return 0; fi
        printf '%s: %s is the origin/main tip and its first parent is not fetched, so the only base would be %s itself; refused (never the tree against itself). Deepen the checkout: git fetch --deepen=1 origin +refs/heads/main:refs/remotes/origin/main\n' "$PROG" "$head" "$head" >&2
        return 1
    fi
    mb=""
    if [ "${ROADMAP_DIFF_FORCE_SHALLOW:-0}" != 1 ]; then
        mb=$(git -C "$REPO_ROOT" merge-base origin/main "$head" 2>/dev/null || true)
    fi
    if [ -n "$mb" ]; then BASE_REF="$mb"; BASE_HOW="merge-base(origin/main, $head)"; return 0; fi
    # read the parents off the commit OBJECT: in a depth-1 clone `rev-list --parents` shows none (shallow graft), `cat-file -p` still does
    parents=$(git -C "$REPO_ROOT" cat-file -p "$head^{commit}" 2>/dev/null | awk '/^parent /{printf "%s ", $2} /^$/{exit}')
    local p1; p1=$(printf '%s\n' "$parents" | cut -d' ' -f1)
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
