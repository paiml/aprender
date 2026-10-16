#!/usr/bin/env bash
# pr_touched_paths.sh -- the paths a pull request touched, for the test-tier decision (#5391).
#
# The tier step used to run `git diff origin/<base> HEAD`: the PR's tip against the
# base branch's CURRENT tip. Every commit that landed on main after the PR branched
# was then counted as the PR's own change (#5358 @ebb62df0a1: 404 paths, and a crate
# only main's #5372 had touched was selected). The PR's change is the diff from
# merge-base(<base>, HEAD).
#
#   bash scripts/ci/pr_touched_paths.sh <base-ref> [<head-ref>]   # paths on stdout, one per line
#
# Prints `touched (merge-base <sha10>): N path(s)` on stderr. When no merge-base can be
# found (a shallow clone that never reached it) it falls back to the tip-vs-tip diff,
# which is a superset of the PR's change: it can over-select, never under-select. It
# says so on stderr instead of silently paying the hour.
# --no-renames (#3664): a rename must list BOTH paths, or a source moved into docs/
# reads as docs-only and runs zero tests.
set -euo pipefail
base="${1:?usage: pr_touched_paths.sh <base-ref> [<head-ref>]}"
head="${2:-HEAD}"
mb=""
if mb=$(git merge-base "$base" "$head" 2>/dev/null); then :; else
  mb=""
  # shallow clone: deepen both sides a bounded number of times, then give up
  for _ in 1 2 3 4; do
    sh=$(git rev-parse --is-shallow-repository 2>/dev/null); [ "$sh" = true ] || break
    git fetch -q --no-tags --deepen=100 origin 2>/dev/null || break
    if mb=$(git merge-base "$base" "$head" 2>/dev/null); then break; fi
    mb=""
  done
fi
if [ -n "$mb" ]; then
  out=$(git diff --no-renames --name-only "$mb" "$head")
  label="merge-base $(printf '%s' "$mb" | cut -c1-10)"
else
  printf '::warning::no merge-base for %s and %s -- the tier falls back to tip vs tip (superset)\n' "$base" "$head" >&2
  out=$(git diff --no-renames --name-only "$base" "$head")
  label="tip vs tip, no merge-base"
fi
[ -z "$out" ] || printf '%s\n' "$out"
printf 'touched (%s): %s path(s)\n' "$label" "$(printf '%s' "$out" | grep -c . || true)" >&2
