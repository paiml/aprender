#!/usr/bin/env bash
# publish_tagged.sh -- publish a version that is ALREADY TAGGED (aprender#6886). Called by release.yml (publish_only).
#   publish_tagged.sh <X.Y.Z> [dry_run 0|1]
# The hourly promote timer tags a green nightly sha and publishes nothing. This is the publish leg for that tag:
# tag commit -> the night's CRUX receipts for exactly that commit (branch nightly-evidence) -> cascade publish.
# It never bumps, tags, pushes or moves a ref. The gates are not here: the tagged commit's own
# scripts/cascade-publish.sh refuses unless clean-room is green on exactly the tag commit and the publish
# preflight (R1-R7) passes. This script only hands R7 the receipts, after checking they are that commit's.
# Fail-closed: no tag, no bundle, a bundle for another commit or version, a checksum that does not hold, a
# verdict that is not green, or no committed prompt certification means nothing is published.
# Seams for the case table: PUB_TAGGED_{REMOTE,PUBLISH,TMP}.
set -uo pipefail
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }  # bashrs disable-line=DET002
die() { say "STOP $*"; exit 1; }
# One function, called on the last line: bash has read the whole file before the checkout below swaps it.
main() {
  local V="${1:-}" DRY="${2:-0}" T REMOTE EVBR PUBLISH SHA B C f want   # TMP is global: the EXIT trap reads it after main returns
  [[ $V =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "STOP bad version '$V' (usage: publish_tagged.sh <X.Y.Z> [dry_run 0|1])" >&2; exit 2; }
  T="v$V"; REMOTE="${PUB_TAGGED_REMOTE:-origin}"; EVBR=nightly-evidence
  PUBLISH="${PUB_TAGGED_PUBLISH:-bash scripts/cascade-publish.sh}"
  [ -z "$(git status --porcelain 2>/dev/null)" ] || die "the tree is not clean; nothing published"
  # No --force: a local tag that differs from the remote's is a refusal, never a move.
  git fetch -q "$REMOTE" "refs/tags/$T:refs/tags/$T" || die "tag $T is not on $REMOTE (or differs from the local tag); nothing published"
  SHA=$(git rev-parse --verify --quiet "refs/tags/$T^{commit}") || die "tag $T does not resolve to a commit"
  git fetch -q "$REMOTE" "+refs/heads/$EVBR:refs/remotes/$REMOTE/$EVBR" || die "cannot fetch $EVBR from $REMOTE"
  B="models-crux/$SHA"
  git cat-file -e "$REMOTE/$EVBR:$B/verdict" 2>/dev/null || die "no CRUX bundle for $SHA on $EVBR ($B/verdict absent)"
  TMP=$(mktemp -d -p "${PUB_TAGGED_TMP:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}" pubtag.XXXXXX) || die "no scratch dir"
  trap 'rm -rf -- "${TMP:?}"' EXIT
  C="$TMP/$B"
  install -d "$C"
  [ -d "$C" ] || die "no scratch dir for the bundle"
  while IFS= read -r f; do   # blobs of the bundle dir only, one by one: no archive is unpacked
    [ "$(git cat-file -t "$REMOTE/$EVBR:$B/$f" 2>/dev/null)" = blob ] || die "bundle $B: $f is not a file"
    git show "$REMOTE/$EVBR:$B/$f" > "$C/$f" || die "cannot read $B/$f"
  done < <(git ls-tree --name-only "$REMOTE/$EVBR:$B")
  [ -s "$C/SHA256SUMS" ] || die "bundle $B has no SHA256SUMS"
  (cd "$C" && sha256sum --check --strict --quiet SHA256SUMS) || die "bundle $B: SHA256SUMS does not hold"
  for f in "$C"/*; do
    f=${f##*/}; [ "$f" = SHA256SUMS ] && continue
    grep -q -F -x -e "$f" <(sed 's/^[0-9a-f]\{64\}  //' "$C/SHA256SUMS") || die "bundle $B: $f is not covered by SHA256SUMS"
  done
  for want in "commit=$SHA" "version=$V" "measure=crux" "state=green"; do
    grep -q -F -x -e "$want" "$C/verdict" || die "bundle $B: verdict has no line '$want'"
  done
  say "CRUX bundle $B: checksums hold, verdict green for $V at $SHA"
  git checkout -q --detach "$SHA" || die "cannot check out $SHA"
  [ "$(git rev-parse HEAD)" = "$SHA" ] || die "HEAD is not $SHA after the checkout"
  [ -s "evidence/crux/$V/prompt-certification.json" ] || die "the tagged commit carries no evidence/crux/$V/prompt-certification.json"
  if [ "$DRY" = 1 ]; then
    say "WOULD publish $T at $SHA: $PUBLISH (clean-room and preflight are asked there); nothing uploaded"; exit 0
  fi
  export MODEL_LADDER_CRUX_DIR="$C" CRUX_CERT="$PWD/evidence/crux/$V/prompt-certification.json"
  $PUBLISH || die "publish refused or failed for $T at $SHA (crates.io is the truth; a rerun resumes)"
  say "PUBLISHED $T at $SHA"
}
main "$@"
