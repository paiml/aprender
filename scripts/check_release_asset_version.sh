#!/usr/bin/env bash
# check_release_asset_version.sh -- the PR-time half of #4275: run the case table of
# scripts/release/asset_version_check.sh, the verdict smoke-cpu and smoke-cuda in
# binary-release.yml pass on every release asset. The asset's `apr --version` must equal
# the tag EXACTLY, -rc.N included (operator 2026-09-24), and its sha must be the tag's
# commit.
#
# That script ships a --self-test, and until this file nothing ran it. It lives in
# scripts/release/, while guard_tree.sh's universe is `git ls-files 'scripts/check_*.sh'`
# and check_guards_are_wired.sh scans scripts/*.sh at depth 1. So the table was
# reachable only by hand, and a regression would first show on a real tag's
# binary-release run -- the failure #4275 was filed from (run 35989369518).
#
# Build-tool free, so guard_tree.sh --no-cargo dispatches it on every PR.
#
#   bash scripts/check_release_asset_version.sh
#
# Exit: 0 when the table passes, 1 when it fails or the script is missing.

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GATE="$ROOT/scripts/release/asset_version_check.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,18p' "$0"
    exit 0
fi

if [[ ! -f "$GATE" ]]; then
    printf 'FAIL: %s is missing -- the table this guard runs was deleted\n' "$GATE"
    exit 1
fi

bash "$GATE" --self-test
rc=$?
if [[ "$rc" -eq 0 ]]; then
    printf 'check_release_asset_version: PASS\n'
    exit 0
fi
printf 'check_release_asset_version: FAIL (asset_version_check --self-test rc %s)\n' "$rc"
exit 1
