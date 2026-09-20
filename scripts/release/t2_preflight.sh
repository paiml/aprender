#!/usr/bin/env bash
# t2_preflight.sh — APR-RELEASE-001 §4 T-0 (operator 2026-09-17): the pre-publish dogfood runs on origin/main HEAD
# BEFORE the bump PR opens; writes preflight-<sha>.verdict = "GO <sha>" or "NO-GO <sha>". prepare_bump.sh --ship refuses without GO.
set -uo pipefail
AP=/mnt/nvme-raid0/agent-wt/rel-0682-autopilot; export PATH=/home/noah/.cargo/bin:$PATH
cd /home/noah/src/aprender || exit 2; git fetch -q origin main || exit 2
sha=$(git rev-parse origin/main); wt="$AP/preflight-wt"
[ -d "$wt" ] && git worktree remove --force "$wt" > /dev/null 2>&1
git worktree add --detach "$wt" "$sha" > /dev/null 2>&1 || exit 2
cd "$wt" || exit 2
export CARGO_TARGET_DIR=/home/noah/src/aprender/target
bash scripts/dogfood.sh --phase pre-publish > "$AP/preflight-$sha.log" 2>&1; rc=$?
# version-unpublished is the one row that legitimately differs pre-bump (the version IS published); everything else must be green
fails=$(grep -E '^\s*\[FAIL\]' "$AP/preflight-$sha.log" | grep -vcE 'version-unpublished' || true)
if [ "$rc" -eq 0 ] || [ "$fails" -eq 0 ]; then printf 'GO %s dogfood_rc=%s fails_excluding_version_row=%s\n' "$sha" "$rc" "$fails" > "$AP/preflight-$sha.verdict"; else printf 'NO-GO %s dogfood_rc=%s fails=%s\n' "$sha" "$rc" "$fails" > "$AP/preflight-$sha.verdict"; fi
cat "$AP/preflight-$sha.verdict"; grep -E '^\s*\[FAIL\]' "$AP/preflight-$sha.log" | cut -c1-160
