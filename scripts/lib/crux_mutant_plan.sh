# shellcheck shell=bash
# crux_mutant_plan.sh — sourced by scripts/check_crux_inference_judge.sh (#4098). Option-neutral: no `set` here; a
# failure is the return status.
#
# crux_mutant_changed <repo root> <out file>: the paths HEAD changes against its base, one per line, into <out>.
# Returns 1 when no base can be named — the caller then says the diff was unreadable, never "untouched".
#
# The base comes from scripts/lib/resolve_base.sh, the project's one resolver for every CI checkout shape: the
# merge-base with origin/main; the merge commit's first parent on a depth-1 pull_request checkout; the queue
# head's parent on a merge_group; the FIRST PARENT on a push to main. GITHUB_BASE_REF alone — what this table
# used first — exists only on pull_request, so a merge group or a push that touched the judge sampled instead of
# running every mutant (quorum round 1, lane 1, measured). CRUX_MUTANT_DIFF_BASE=<ref> overrides (<ref>...HEAD).
crux_mutant_changed() {
  local root=$1 out=$2
  if [ -n "${CRUX_MUTANT_DIFF_BASE:-}" ]; then
    # A base whose merge-base with HEAD is HEAD itself (HEAD, or anything ahead of it) diffs the tree against itself:
    # empty, and read as "untouched". Refused like resolve_base refuses it (quorum round 3, lane 2, measured).
    local mb head
    mb=$(git -C "$root" merge-base "$CRUX_MUTANT_DIFF_BASE" HEAD 2> /dev/null) || return 1
    head=$(git -C "$root" rev-parse HEAD 2> /dev/null) || return 1
    [ "$mb" != "$head" ] || return 1
    git -C "$root" diff --no-renames --name-only "$mb" HEAD > "$out" 2> /dev/null
    return
  fi
  # In a SUBSHELL: resolve_base needs REPO_ROOT and PROG, and setting them here clobbered the sourcing table's own
  # PROG, so every real run printed another guard's name in its summary (quorum round 2, lane 1, measured).
  local base
  base=$(
    REPO_ROOT=$root
    PROG=crux_mutant_plan
    # shellcheck source=scripts/lib/resolve_base.sh
    . "$root/scripts/lib/resolve_base.sh" || exit 1
    BASE_REF=""
    resolve_base HEAD 2> /dev/null || exit 1
    printf '%s' "$BASE_REF"
  ) || return 1
  [ -n "$base" ] || return 1
  # --no-renames: a rename lists BOTH paths, so a watched file moved away still counts as touched (#3664)
  git -C "$root" diff --no-renames --name-only "$base" HEAD > "$out" 2> /dev/null
}
