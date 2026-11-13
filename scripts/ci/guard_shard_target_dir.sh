#!/usr/bin/env bash
# guard_shard_target_dir.sh - one HOST cargo target dir per guard-cargo shard (#7026).
#
# The x86 fat job runs the guard-cargo-shard sections at once on ONE runner, and
# fat_driver.py hands every section the runner's environment, whose systemd drop-in
# exports CARGO_TARGET_DIR=/var/lib/ci-target/<runner>. Docker steps mount their own
# per-shard GUARD_TARGET_DIR; host steps did not. So contracts_gate.sh -> pv_bin.sh in
# every shard built the same <target>/debug/pv, the shard that built last owned it, and
# pv_bin.sh rightly refused it in the others as WRONG-TREE (main run 43430038433
# attempt 1, guard-cargo#m30 rc=3; attempt 2 green: a race, not a code defect).
#
# Prints ONE line for GITHUB_ENV, `CARGO_TARGET_DIR=<base>/guard-shard-<N>`, and creates
# the dir. <base> is the inherited CARGO_TARGET_DIR (warm across runs, like
# <target>/facades), else $GITHUB_WORKSPACE/target.
#
#   bash scripts/ci/guard_shard_target_dir.sh <shard> >> "$GITHUB_ENV"
#
# Falsifier: scripts/check_guard_shard_target_dirs.sh.
# EXIT 0 printed; 2 the shard is not a positive integer, or no base dir can be named.
set -uo pipefail

shard=${1:-}
case "$shard" in
    '' | *[!0-9]* | 0*)
        echo "guard_shard_target_dir: shard must be a positive integer, got '${shard}'" >&2
        exit 2
        ;;
esac
base=${CARGO_TARGET_DIR:-}
[ -n "$base" ] || base=${GITHUB_WORKSPACE:+$GITHUB_WORKSPACE/target}
[ -n "$base" ] || { echo "guard_shard_target_dir: neither CARGO_TARGET_DIR nor GITHUB_WORKSPACE is set" >&2; exit 2; }
base=${base%/}
# A base that is already a shard's dir (this step re-run in one job) is not nested again.
case "${base##*/}" in guard-shard-*) base=${base%/*} ;; esac
dir="$base/guard-shard-$shard"
mkdir -p -- "$dir" || { echo "guard_shard_target_dir: cannot create $dir" >&2; exit 2; }
printf 'CARGO_TARGET_DIR=%s\n' "$dir"
