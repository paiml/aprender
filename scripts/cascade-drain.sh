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

# THE UNIVERSE IS READ FROM CARGO, NOT FROM THE CASCADE'S OWN TABLE.
#
# This used to be `grep -oE 'TIERS\[...' scripts/cascade-publish.sh`, which made
# the drain's idea of "the release" a copy of the cascade's. A crate absent from
# TIERS[] was therefore absent from TOTAL as well, so it was never published,
# never counted, and never missed: the drain printed DRAIN COMPLETE n/n while
# the three crates/facades crates sat unshipped (aprender#2559). Two checks
# sharing one blind list is one check.
#
# Reading from `cargo metadata` across every workspace inverts that: if the
# cascade under-covers, TOTAL exceeds what any pass can publish and the drain
# says STUCK — loudly, instead of silently agreeing.
#
# Each row also carries its OWN expected version. The facades version
# independently of the aprender line (0.4.0 vs 0.63.0, aprender#2546); a single
# $TARGET would mark them permanently behind, and the drain would exit 2 STUCK
# on an append-only registry — the most dangerous thing this script can report.
UNIVERSE=$(python3 scripts/lib/cascade_universe.py "$REPO_ROOT") || {
    echo "ERROR: cascade_universe.py failed; refusing to drain against an unknown universe" >&2
    exit 2
}
# name<TAB>expected-version. Root-workspace crates are pinned to $TARGET so that
# an explicit --target still overrides what cargo read from the manifests;
# crates from any other workspace keep their own version.
WANT=$(printf '%s\n' "$UNIVERSE" | REPO_ROOT="$REPO_ROOT" TARGET="$TARGET" awk -F'\t' '
    { print $1 "\t" ($4 == ENVIRON["REPO_ROOT"] ? ENVIRON["TARGET"] : $2) }' | sort -u)
TOTAL=$(printf '%s\n' "$WANT" | grep -c .)
[ "$TOTAL" -gt 0 ] || { echo "ERROR: no crates in the publish universe" >&2; exit 2; }
# Vacuity: a shrunken universe would let the drain report DRAIN COMPLETE over
# almost nothing. 70 is the root workspace alone; anything less is a broken
# enumeration, not a smaller repo.
[ "$TOTAL" -ge 70 ] || { echo "ERROR vacuity: universe holds $TOTAL crates, expected 70 or more" >&2; exit 2; }

# Count how many tiered crates are live on crates.io at $TARGET.
#
# Uses the SPARSE INDEX (index.crates.io), never the JSON API. During the v0.61.0
# cascade the API rate-limited us: 12-way parallel polling every pass made
# /api/v1/crates/<name> return EMPTY for all 70 crates. count_pub then read 0,
# and the drain aborted with "STUCK: no progress (58 -> 0)/70" while 60 crates
# were in fact published — a false failure on an append-only registry, which is
# the single most dangerous thing this script can report (see the STUCK guard
# below: the documented response to STUCK is "this is not a transient"). The
# sparse index is a static CDN, is what cargo itself resolves against, and is not
# subject to that rate limit.
#
# Index path layout: 1-char -> 1/n, 2-char -> 2/n, 3-char -> 3/<c1>/n,
# else <c1c2>/<c3c4>/n. The last NDJSON line is the newest version.
#
# UA/TARGET pass via the environment rather than being spliced into the
# single-quoted body: the old '"$UA"' splicing was unparseable (bashrs SC1078)
# and broke on any character needing quoting.
# Each line of $WANT is "<crate>\t<expected version>", so the worker compares
# against the version THAT crate is supposed to reach rather than one global
# target. Passing the pair through xargs keeps the single-quoted body free of
# splicing (the SC1078 hazard the UA/TARGET env vars already document).
count_pub() {
    printf '%s\n' "$WANT" | DRAIN_UA="$UA" xargs -P 8 -I{} bash -c '
        IFS="$(printf "\t")" read -r crate DRAIN_TARGET <<< "$1"
        n=${#crate}
        if   [ "$n" -eq 1 ]; then p="1/${crate}"
        elif [ "$n" -eq 2 ]; then p="2/${crate}"
        elif [ "$n" -eq 3 ]; then p="3/${crate:0:1}/${crate}"
        else p="${crate:0:2}/${crate:2:2}/${crate}"
        fi
        v=$(curl -s --retry 3 --retry-delay 2 -H "User-Agent: ${DRAIN_UA}" "https://index.crates.io/${p}" 2>/dev/null \
            | grep -v "^[[:space:]]*$" | tail -1 | sed -n "s/.*\"vers\":\"\([^\"]*\)\".*/\1/p")
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
        | grep -E "PUBLISHED|FATAL|DEFER|already ${TARGET}|TIER|RETRY" | tail -60

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
