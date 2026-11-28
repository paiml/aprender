#!/usr/bin/env bash
# Previous final tag for a release window: the newest vX.Y.Z strictly below the target
# that is an ancestor of the cut. Same rule as .github/workflows/release.yml (T-5 gate).
# usage: prev_final_tag.sh <target-version> <cut-ref>   prints the version without v; rc 2 if none
set -uo pipefail
target=${1:?target version}
cut=${2:?cut ref}
candidates=$(git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sed 's/^v//' | grep -vxF "$target" \
  | { sort -V; printf '%s\n' "$target"; } | sort -V | awk -v v="$target" '$0==v{exit} {print}' | sort -rV)
while IFS= read -r t; do
  [ -n "$t" ] || continue
  git merge-base --is-ancestor "v$t" "$cut" && { echo "$t"; exit 0; }
done <<< "$candidates"
exit 2
