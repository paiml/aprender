#!/usr/bin/env bash
# advisory_pin_bump.sh -- P9 F7: advance ci/advisory-db.pin to the live RustSec HEAD and open the
# bump PR. Run daily from a fleet timer as an EXISTING gh identity (a PR opened with the Actions
# GITHUB_TOKEN never triggers CI). If this stops landing, advisory_pin.sh age goes red on main.
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
git clone -q --depth 1 https://github.com/RustSec/advisory-db.git "$work/db"
sha=$(git -C "$work/db" rev-parse HEAD)
d=$(git -C "$work/db" log -1 --format=%cs)
if grep -q "^sha=$sha\$" "$repo_root/ci/advisory-db.pin"; then echo "pin already at $sha"; exit 0; fi
git -C "$repo_root" fetch -q origin main
br="chore/advisory-pin-$d"
git -C "$repo_root" worktree add -q -b "$br" "$work/wt" origin/main
printf 'sha=%s\ndate=%s\n' "$sha" "$d" > "$work/wt/ci/advisory-db.pin"
git -C "$work/wt" commit -qam "chore(advisory): bump pinned RustSec snapshot to $d"
git -C "$work/wt" push -q origin "$br"
(cd "$work/wt" && gh pr create --base main --head "$br" --title "chore(advisory): bump pinned RustSec snapshot to $d" \
    --body "Daily pin bump (P9 option 4, F7). Owner: flow." && gh pr merge "$br" --squash --auto --delete-branch)
