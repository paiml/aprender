#!/usr/bin/env bash
# One release at a time (#5561). Exit 0 = this caller owns the release, 1 = refuse, 2 = not measured (refuse).
#   release_exclusive.sh <vX.Y.Z> [self_run_id]      claim the version (release.yml `plan`, autopilot.sh)
#   release_exclusive.sh --release <vX.Y.Z>          drop a claim (release.yml `summary` after a failed publish)
# Refuses when a release or draft, or a tag, named <vX.Y.Z> exists; when another release.yml run is
# active (ONE run-list snapshot, every status but `completed`; in `plan` only a LOWER run id blocks, so
# two rcs dispatched together never both stop); and when the claim ref refs/release-claims/<vX.Y.Z>
# exists. The claim is created last and atomically (the API refuses a ref that exists), so a caller
# that passed the checks while another did too loses at the claim. autopilot.sh passes no run id.
# Only a 404 reads as "absent": any other gh error is exit 2, never a green.
set -euo pipefail
repo=${GITHUB_REPOSITORY:-${REPO:?GITHUB_REPOSITORY or REPO is required}}
if [ "${1:-}" = --release ]; then
  gh api -X DELETE "repos/$repo/git/refs/release-claims/${2:?usage: --release vX.Y.Z}" > /dev/null 2>&1 || true
  exit 0
fi
tag=${1:?usage: release_exclusive.sh vX.Y.Z [self_run_id]}
self=${2:-}

# absent <label> <cmd...>: 0 = the object is absent (404), 1 = it exists, 2 = could not tell.
absent() {
  local label=$1 out; shift
  if out=$("$@" 2>&1); then echo "release_exclusive: $label exists" >&2; return 1; fi
  case $out in *"Not Found"*|*"not found"*|*"HTTP 404"*) return 0 ;; esac
  echo "release_exclusive: $label not measured: ${out:0:200}" >&2; return 2
}
rc=0; absent "release or draft $tag" gh release view "$tag" --repo "$repo" || rc=$?
[ "$rc" = 0 ] || exit "$rc"
absent "tag $tag" gh api "repos/$repo/git/ref/tags/$tag" || rc=$?
[ "$rc" = 0 ] || exit "$rc"

ids=$(gh run list --repo "$repo" --workflow release.yml --limit 100 --json databaseId,status \
  --jq '.[] | select(.status != "completed") | .databaseId') \
  || { echo "release_exclusive: run list not measured" >&2; exit 2; }
for id in $ids; do
  [ -n "$self" ] && [ "$id" -ge "$self" ] && continue
  echo "release_exclusive: release.yml run $id is active" >&2; exit 1
done

sha=$(gh api "repos/$repo/commits/HEAD" -q .sha 2>/dev/null) || { echo "release_exclusive: cannot resolve HEAD for the claim" >&2; exit 2; }
if ! out=$(gh api -X POST "repos/$repo/git/refs" -f "ref=refs/release-claims/$tag" -f "sha=$sha" 2>&1); then
  case $out in
    *"already exists"*|*"Reference already exists"*) echo "release_exclusive: $tag is claimed by another release" >&2; exit 1 ;;
  esac
  echo "release_exclusive: claim not measured: ${out:0:200}" >&2; exit 2
fi
echo "release_exclusive: $tag claimed"
