#!/usr/bin/env bash
# one_command.sh -- one-command release (KZ-001 b, aprender#6729). Called by .github/workflows/release.yml.
#   one_command.sh <X.Y.Z> [dry_run 0|1] <green-nightly-sha>   (T5: arg 3 is required; never main HEAD)
# bump on release/X.Y.Z -> clean-room (infra) on THAT sha -> tag that sha -> publish. Fail-closed: no verdict,
# a red verdict, or a verdict on another sha means no tag and no publish. dry_run stops after the bump (WOULD lines).
# Never --force, never --no-verify, no tag delete or move. Seams for the case table: ONE_CMD_{BUMP,ARM,PUBLISH,POLL}.
set -uo pipefail
V="${1:?usage: one_command.sh <X.Y.Z> [dry_run 0|1]}"; DRY="${2:-0}"
[[ $V =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "STOP bad version '$V'" >&2; exit 2; }
INFRA="${ONE_CMD_INFRA:-paiml/infra}"; POLL="${ONE_CMD_POLL:-300}"; T="v$V"; BR="release/$V"
BUMP="${ONE_CMD_BUMP:-bash scripts/bump-version.sh}"; PUBLISH="${ONE_CMD_PUBLISH:-bash scripts/cascade-publish.sh}"
ARM="${ONE_CMD_ARM:-bash scripts/hooks/pre-push-tags.sh --arm-release}"
# PMAT-5306 seams: CERT carries the CRUX certification, MODELS measures CRUX smoke, REACH proves both hosts first.
CERT="${ONE_CMD_CERT:-bash scripts/release/prepare_bump.sh --carry-cert}"; MODELS="${ONE_CMD_MODELS:-bash scripts/release/models_t1.sh}"
oc_reach() { local h; h=$(hostname -s) || return 1; [[ ${h,,} == *lambda* ]] && ssh -o BatchMode=yes -o ConnectTimeout=10 gx10 true; }
REACH="${ONE_CMD_REACH:-oc_reach}"
MOUT="${ONE_CMD_MODELS_OUT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/models-t1-$V}"; CC="$PWD/evidence/crux/$V/prompt-certification.json"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }  # bashrs disable-line=DET002
die() { say "STOP $*"; exit 1; }
# APR-RELEASE-TRAIN T5: the release builds from the green nightly sha it is handed (arg 3), never main HEAD, so
# feature PRs keep merging while it runs. NEXTDEV opens the X.(Y+1).0-dev bump on main the moment the tag is pushed.
BASE="${3:-}"; NEXTDEV="${ONE_CMD_NEXTDEV:-bash scripts/release/next_dev_bump.sh}"
[[ $BASE =~ ^[0-9a-f]{40}$ ]] || { echo "STOP no green nightly sha (arg 3, 40 hex); a release never builds from main HEAD" >&2; exit 2; }
git merge-base --is-ancestor "$BASE" origin/main || die "$BASE is not on origin/main"
git checkout -q -b "$BR" "$BASE" || die "cannot create $BR at $BASE"
$BUMP "$V" || die "bump $V failed"
# PMAT-5306: R7 (check_publish_preflight.sh) judges CRUX smoke against evidence/crux/<V>/prompt-certification.json at
# the tagged tree, so the certification is carried into the bump commit (prepare_bump's pb_carry_cert), never after.
$CERT "$PWD" "$V" "$BASE" || die "the CRUX prompt certification for $V could not be carried; no tag"
git add -- "evidence/crux/$V" || die "cannot stage evidence/crux/$V"
git commit -q -am "chore(release): bump to $V" || die "bump commit failed"
SHA=$(git rev-parse HEAD) || die "no sha"
git cat-file -e "$SHA:evidence/crux/$V/prompt-certification.json" || die "bump commit $SHA holds no evidence/crux/$V/prompt-certification.json; no tag"
say "BUMP $V at $SHA on $BR"
if [ "$DRY" = 1 ]; then
  say "WOULD push $BR; models_t1 crux at $SHA; clean-room ref=$SHA; git tag -a $T $SHA; push $T; MODEL_LADDER_CRUX_DIR=$MOUT CRUX_CERT=$CC $PUBLISH; carry PR to main"; exit 0
fi
# PMAT-5306: the CRUX smoke at THIS sha on lambda and gx10 before any tag; R7 re-judges these receipts at publish.
# A host that cannot be reached is not_measured, and not_measured is a stop: nothing is skipped or stood in for.
# Reachability is proven before the release branch is pushed, so an unreachable host leaves no release/V behind.
$REACH || die "not_measured: CRUX smoke needs this runner to be lambda and gx10 reachable over SSH; no tag"
git push -q origin "$BR" || die "push $BR failed"
MODELS_T1_MEASURE=crux $MODELS "$V" "$SHA" "$MOUT" > "$MOUT.log" 2>&1; mrc=$?
grep -E '^MODELS ' "$MOUT.log"
[ "$mrc" = 0 ] && grep -q '^MODELS GO (CRUX smoke) ' "$MOUT.log" \
  || die "CRUX smoke at $SHA is not GO (models_t1 rc $mrc; rc 2 = not_measured, a stop); no tag ($MOUT.log)"
say "MODELS GO (CRUX smoke) at $SHA"
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
late=0; $NEXTDEV "$V" || { say "ANDON next-dev bump after $T failed; publish continues"; late=1; }
MODEL_LADDER_CRUX_DIR="$MOUT" CRUX_CERT="$CC" $PUBLISH || die "publish failed after tag $T (tag stays; crates.io is the truth)"
say "PUBLISHED $T"
[ "$late" = 0 ] || die "published $T, but the next-dev bump PR was not opened"
