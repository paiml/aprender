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
    git -C "$root" diff --no-renames --name-only "$CRUX_MUTANT_DIFF_BASE...HEAD" > "$out" 2> /dev/null
    return
  fi
  REPO_ROOT=$root
  PROG=crux_mutant_plan
  # shellcheck source=scripts/lib/resolve_base.sh
  . "$root/scripts/lib/resolve_base.sh" || return 1
  BASE_REF=""
  resolve_base HEAD 2> /dev/null || return 1
  [ -n "$BASE_REF" ] || return 1
  # --no-renames: a rename lists BOTH paths, so a watched file moved away still counts as touched (#3664)
  git -C "$root" diff --no-renames --name-only "$BASE_REF" HEAD > "$out" 2> /dev/null
}
