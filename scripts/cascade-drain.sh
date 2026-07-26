#!/usr/bin/env bash
# Auto-drain the crates.io publish cascade.
#
# Loops `scripts/cascade-publish.sh` until every crate named in its TIERS[]
# arrays is live on crates.io at the target version. Each pass resolves one
# more dependency layer: a crate whose sibling dep was published in pass N
# becomes publishable in pass N+1. A single pass is NOT sufficient — the
# v0.60.0 cascade needed a multi-pass drain, which is why this exists.
#
# Was previously an out-of-tree scratch script (aprender-worktrees/cascade_drain.sh)
# with TARGET=0.60.0 and the worktree path hardcoded. Running it unmodified for a
# later release silently drains to the WRONG version and reports success, so it
# now lives in-tree, defaults the target to the workspace version, and accepts
# --target.
#
# Usage:
#   scripts/cascade-drain.sh                  # target = workspace version
#   scripts/cascade-drain.sh --target 0.61.0
#   scripts/cascade-drain.sh --passes 30    # default is 20
#
# Exit codes: 0 drained | 2 stuck (no progress) | 3 passes exhausted
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT" || exit 2

TARGET=""
# 20, not 6. TIERS[] in cascade-publish.sh is NOT a topological order — there are
# 44 dependency edges pointing into the same or a later tier (e.g. aprender-core
# in T2 depends on aprender-data in T8; apr-cli T10 -> aprender-registry T13).
# cascade-publish.sh does exactly ONE retry round and then exits 1, so each drain
# pass only resolves roughly one dependency layer. Simulation against the real
# dep graph converges in ~15 passes; the old default of 6 exhausted first and
# exited 3, which reads as a FAILED release while the cascade was in fact still
# making forward progress. Cheap to over-provision: a pass with nothing left to
# publish is one crates.io census (~seconds), and the loop exits 0 the moment it
# reaches N/N.
PASSES=20
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET="${2:-}"; shift 2 ;;
        --passes) PASSES="${2:-20}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Default to the workspace version — the same source cascade-publish.sh uses,
# so the two can never disagree about what "the release" means.
if [ -z "$TARGET" ]; then
    TARGET=$(grep -E '^version = "' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
fi
[ -n "$TARGET" ] || { echo "ERROR: could not determine target version" >&2; exit 2; }

# crates.io rejects requests without a User-Agent; without one check_version
# returns empty for EVERY crate and the drain reports 0/N forever.
UA="aprender-release/${TARGET} (noah.gift@gmail.com)"

CRATES=$(grep -oE 'TIERS\[[0-9]+\]="[^"]*"' scripts/cascade-publish.sh \
         | sed 's/.*="//;s/"//' | tr ' ' '\n' | sort -u)
TOTAL=$(printf '%s\n' "$CRATES" | grep -c .)
[ "$TOTAL" -gt 0 ] || { echo "ERROR: no crates parsed from TIERS[]" >&2; exit 2; }

# Count how many tiered crates are live on crates.io at $TARGET.
# UA/TARGET are passed via the environment rather than spliced into the
# single-quoted body: the old '"$UA"' splicing was unparseable (bashrs SC1078)
# and broke on any character needing quoting. max_version is extracted with
# sed so this needs no python in the hot loop.
count_pub() {
    printf '%s\n' "$CRATES" | DRAIN_UA="$UA" DRAIN_TARGET="$TARGET" xargs -P 12 -I{} bash -c '
        crate="$1"
        v=$(curl -s -H "User-Agent: ${DRAIN_UA}" "https://crates.io/api/v1/crates/${crate}" 2>/dev/null | sed -n "s/.*\"max_version\":\"\([^\"]*\)\".*/\1/p")
        [ "$v" = "${DRAIN_TARGET}" ] && echo ok
    ' _ {} | grep -c ok
}

echo "=== cascade drain: target=$TARGET crates=$TOTAL passes=$PASSES ==="

# Never race a running cascade — two concurrent publishes of the same crate
# produce confusing 403/409s that look like auth failures.
CASCADE_PROC="scripts/cascade-publish.sh"
if pgrep -f "$CASCADE_PROC" >/dev/null 2>&1; then
    echo "waiting for the running cascade to finish..."
    while pgrep -f "$CASCADE_PROC" >/dev/null 2>&1; do
        sleep 30
    done
fi

# A dev [patch.crates-io] section makes `cargo publish` resolve siblings to
# local paths and fail verification. Safe to drop: it is untracked dev config.
rm -f "$REPO_ROOT/.cargo/config.toml"

for pass in $(seq 1 "$PASSES"); do
    n=$(count_pub)
    echo "=== PASS $pass START: $n/$TOTAL at $TARGET ==="
    if [ "$n" -ge "$TOTAL" ]; then echo "DRAIN COMPLETE: $n/$TOTAL at $TARGET"; exit 0; fi

    # A stale CARGO_REGISTRY_TOKEN 403s every upload; the credentials file wins.
    ( unset CARGO_REGISTRY_TOKEN; bash scripts/cascade-publish.sh ) 2>&1 \
        | grep -E "PUBLISHED|FATAL|already ${TARGET}|TIER|RETRY" | tail -40

    n2=$(count_pub)
    echo "=== PASS $pass END: $n2/$TOTAL ==="
    if [ "$n2" -ge "$TOTAL" ]; then echo "DRAIN COMPLETE: $n2/$TOTAL at $TARGET"; exit 0; fi
    # No forward progress means a genuine publish error, not a transient —
    # read the log rather than re-running. Never bump the version mid-drain.
    if [ "$n2" -le "$n" ]; then
        echo "STUCK: no progress ($n -> $n2)/$TOTAL after pass $pass" >&2
        exit 2
    fi
done

echo "EXHAUSTED $PASSES passes at $(count_pub)/$TOTAL" >&2
exit 3
