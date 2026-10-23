#!/usr/bin/env bash
# advisory_pin_bump.sh -- P9 F7: advance ci/advisory-db.pin to the live RustSec HEAD and open the
# bump PR. Run daily from a fleet timer as an EXISTING gh identity (a PR opened with the Actions
# GITHUB_TOKEN never triggers CI). If this stops landing, advisory_pin.sh age goes red on main.
# Writes verified=<today> EVERY run, even when the sha is unchanged: staleness is the time since the
# pin was last confirmed current, so a quiet RustSec week does not turn F5 red.
# Arming the bump PR is its owner's step (this script never arms). Producer of F7: a daily fleet timer on an infra host (OWNER: infra; NOT YET SCHEDULED, see inbox).
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
git clone -q --depth 1 https://github.com/RustSec/advisory-db.git "$work/db"
sha=$(git -C "$work/db" rev-parse HEAD)
d=$(git -C "$work/db" log -1 --format=%cs)
today=$(date -u +%F)
git -C "$repo_root" fetch -q origin main
br="chore/advisory-pin-$today"
git -C "$repo_root" worktree add -q -b "$br" "$work/wt" origin/main
printf 'sha=%s\ndate=%s\nverified=%s\n' "$sha" "$d" "$today" > "$work/wt/ci/advisory-db.pin"
git -C "$work/wt" commit -qam "chore(advisory): bump pinned RustSec snapshot (verified $today, advisory-db $d)"
git -C "$work/wt" push -q origin "$br"
(cd "$work/wt" && gh pr create --base main --head "$br" --title "chore(advisory): bump pinned RustSec snapshot (verified $today, advisory-db $d)" \
    --body "Daily pin bump (P9 option 4, F7). Owner: flow.")
