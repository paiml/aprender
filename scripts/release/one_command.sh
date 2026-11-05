#!/usr/bin/env bash
# one_command.sh -- one-command release (KZ-001 b, aprender#6729). Called by .github/workflows/release.yml.
#   one_command.sh <X.Y.Z> [dry_run 0|1]
# bump on release/X.Y.Z -> clean-room (infra) on THAT sha -> tag that sha -> publish. Fail-closed: no verdict,
# a red verdict, or a verdict on another sha means no tag and no publish. dry_run stops after the bump (WOULD lines).
# Never --force, never --no-verify, no tag delete or move. Seams for the case table: ONE_CMD_{BUMP,ARM,PUBLISH,POLL}.
set -uo pipefail
V="${1:?usage: one_command.sh <X.Y.Z> [dry_run 0|1]}"; DRY="${2:-0}"
[[ $V =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "STOP bad version '$V'" >&2; exit 2; }
INFRA="${ONE_CMD_INFRA:-paiml/infra}"; POLL="${ONE_CMD_POLL:-300}"; T="v$V"; BR="release/$V"
BUMP="${ONE_CMD_BUMP:-scripts/bump-version.sh}"; PUBLISH="${ONE_CMD_PUBLISH:-scripts/cascade-publish.sh}"
ARM="${ONE_CMD_ARM:-bash scripts/hooks/pre-push-tags.sh --arm-release}"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }  # bashrs disable-line=DET002
die() { say "STOP $*"; exit 1; }
git checkout -q -b "$BR" || die "cannot create $BR"
$BUMP "$V" || die "bump $V failed"
git commit -q -am "chore(release): bump to $V" || die "bump commit failed"
SHA=$(git rev-parse HEAD) || die "no sha"
say "BUMP $V at $SHA on $BR"
if [ "$DRY" = 1 ]; then
  say "WOULD push $BR; clean-room ref=$SHA; git tag -a $T $SHA; push $T; $PUBLISH; carry PR to main"; exit 0
fi
git push -q origin "$BR" || die "push $BR failed"
t0=$(date -u +%s)  # bashrs disable-line=DET002
gh workflow run clean-room.yml --repo "$INFRA" -f repos=aprender -f ref="$SHA" || die "clean-room dispatch failed"
sleep "$POLL"; id=""
for _ in 1 2 3 4 5 6; do
  id=$(gh run list --repo "$INFRA" --workflow clean-room.yml --event workflow_dispatch --limit 10 --json databaseId,createdAt \
       --jq "[.[]|select((.createdAt|fromdateiso8601)>=$t0-30)]|sort_by(.createdAt)|last|.databaseId // empty")
  [ -n "$id" ] && break; sleep "$POLL"
done
[ -n "$id" ] || die "no clean-room run appeared for $SHA"
c=""
for _ in $(seq 1 ${ONE_CMD_TRIES:-3}); do
  c=$(gh run view "$id" --repo "$INFRA" --json jobs --jq '.jobs[]|select(.name=="clean-room (aprender)" and .status=="completed")|.conclusion' | head -1)
  [ -n "$c" ] && break; sleep "$POLL"
done
[ "$c" = success ] || die "clean-room (aprender) run $id concluded '${c:-absent}'; no tag"
tested=$(gh run view "$id" --repo "$INFRA" --log | sed -n 's/.*tested-sha: *\([0-9a-f]\{40\}\)\r*$/\1/p' | head -1)
[ "$tested" = "$SHA" ] || die "clean-room run $id tested '${tested:-nothing}', not $SHA; no tag"
say "CLEANROOM GREEN run $id on $SHA"
git tag -a "$T" -m "aprender $T" "$SHA" || die "tag failed"
$ARM "$T" 120 || die "arming $T failed"
git push origin "$T" || die "tag push failed"
say "TAG $T at $SHA"
$PUBLISH || die "publish failed after tag $T (tag stays; crates.io is the truth)"
say "PUBLISHED $T"
