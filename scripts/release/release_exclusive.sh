#!/usr/bin/env bash
# One release at a time (#5561). Exit 0 = this caller may proceed, 1 = refuse, 2 = not measured (refuse).
#   release_exclusive.sh <vX.Y.Z> [self_run_id]
# Refuses when a release or draft, or a tag, named <vX.Y.Z> exists, or another release.yml run is
# queued or in progress. In release.yml's plan job self_run_id is $GITHUB_RUN_ID and the LOWEST run id
# wins, so two rcs dispatched together never both stop: exactly one proceeds. autopilot.sh passes no
# id, so any active release.yml run stops it.
set -euo pipefail
tag=${1:?usage: release_exclusive.sh vX.Y.Z [self_run_id]}
self=${2:-}
repo=${GITHUB_REPOSITORY:-${REPO:?GITHUB_REPOSITORY or REPO is required}}

if gh release view "$tag" --repo "$repo" > /dev/null 2>&1; then
  echo "release_exclusive: a release or draft $tag already exists" >&2; exit 1
fi
if gh api "repos/$repo/git/ref/tags/$tag" > /dev/null 2>&1; then
  echo "release_exclusive: tag $tag already exists" >&2; exit 1
fi
for st in queued in_progress waiting requested; do
  ids=$(gh run list --repo "$repo" --workflow release.yml --status "$st" --json databaseId \
    --jq '.[].databaseId') || { echo "release_exclusive: run list ($st) not measured" >&2; exit 2; }
  for id in $ids; do
    [ -n "$self" ] && [ "$id" -ge "$self" ] && continue
    echo "release_exclusive: release.yml run $id is $st" >&2; exit 1
  done
done
echo "release_exclusive: $tag is free"
