#!/usr/bin/env bash
# prev_final_tag.sh <version> <cutref> — print the previous final tag for the reconcile window (aprender R2, 2026-11-28).
#
# The previous tag is the newest final vX.Y.Z strictly below <version> that is an ancestor of <cutref>. This is the rule
# .github/workflows/release.yml applies on the release path; the nightly (release-gates-nightly.yml) calls this script so
# the nightly judges the same window. A sort-only pick can name a tag from another release line (v0.75.3 is not an
# ancestor of v0.75.4), and check_reconcile.sh then refuses the window with rc 2.
#
# Exit 0 and the tag name (without the v) on stdout; exit 2 with no stdout when no ancestor tag exists below <version>.
set -uo pipefail

version="${1:?usage: prev_final_tag.sh <version> <cutref>}"
cutref="${2:?usage: prev_final_tag.sh <version> <cutref>}"

candidates=$(git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sed 's/^v//' | grep -vxF "$version" \
  | { sort -V; printf '%s\n' "$version"; } | sort -V | awk -v v="$version" '$0==v{exit} {print}' | sort -rV)

prev=''
while IFS= read -r t; do
  [ -n "$t" ] || continue
  git merge-base --is-ancestor "v$t" "$cutref" && { prev="$t"; break; }
done <<< "$candidates"

[ -n "$prev" ] || { echo "::error::no final ancestor tag below v$version: the reconcile window cannot be named" >&2; exit 2; }
printf '%s\n' "$prev"
