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
# Exit codes: 0 drained AND smoke-verified | 2 stuck (no progress) | 3 passes exhausted
#             4 post-publish smoke verdict YANK (yanked, dry-run, refused or yank failed)
#             5 post-publish smoke verdict NOT_MEASURED (nothing was verified; no yank)
# A yanked version is never counted as published.
#
# DRAIN COMPLETE (exit 0) needs TWO things: every crate live at its version, and
# a `SMOKE_RESULT PASS version=$TARGET` line that THIS drain saw. A crate count
# alone is not a release: a cascade pass can upload the last crate and die before
# its smoke block. When every crate is live and no such line was seen, the drain
# runs `post_publish_smoke.sh --version $TARGET --train-published $TARGET` itself
# and exits on that verdict (PASS 0, YANK 4, anything else 5).
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
        last=$(curl -s --retry 3 --retry-delay 2 -H "User-Agent: ${DRAIN_UA}" "https://index.crates.io/${p}" 2>/dev/null \
            | grep -v "^[[:space:]]*$" | tail -1)
        v=$(printf "%s\n" "$last" | sed -n "s/.*\"vers\":\"\([^\"]*\)\".*/\1/p")
        case "$last" in *\"yanked\":true*) exit 0 ;; esac
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
# local paths and fail verification. It is untracked dev config, so it is MOVED
# into this drain's own mktemp root and MOVED BACK on exit by the trap below,
# never deleted: this script runs no rm at all. If something recreated the file
# meanwhile, the trap leaves both in place and says where the moved copy is.
drain_root=$(mktemp -d "${TMPDIR:-/tmp}/cascade-drain.XXXXXX") || { echo "FATAL: mktemp -d for the drain root failed" >&2; exit 2; }
moved_config="${drain_root:?}/config.toml.moved-aside"
restore_config() {
    [ -e "$moved_config" ] || return 0
    if [ -e "${REPO_ROOT:?}/.cargo/config.toml" ]; then
        echo "NOT restored: ${REPO_ROOT}/.cargo/config.toml exists again; the moved copy stays at $moved_config" >&2
        return 0
    fi
    mv -- "$moved_config" "${REPO_ROOT:?}/.cargo/config.toml" \
        && echo "restored .cargo/config.toml" \
        || echo "NOT restored: mv back failed; the moved copy stays at $moved_config" >&2
}
trap restore_config EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [ -e "${REPO_ROOT:?}/.cargo/config.toml" ]; then
    mv -- "${REPO_ROOT:?}/.cargo/config.toml" "$moved_config"
    echo "moved .cargo/config.toml aside to $moved_config (moved back on exit)"
fi

# The cascade log lives in the drain root and is bounded: each pass writes only
# the LAST $DRAIN_LOG_CAP bytes of the cascade output (the SMOKE_RESULT line is
# the last line, so it survives the cap), and the previous pass's log is rotated
# to cascade.log.prev. At most two capped files exist, whatever the pass count.
DRAIN_LOG_CAP=8388608
cascade_log="$drain_root/cascade.log"
smoke_log="$drain_root/smoke.log"

# Set only by an exact `SMOKE_RESULT PASS version=$TARGET` line in a cascade log
# this drain wrote. Never by a count, an empty line or another version's PASS.
smoke_pass_seen=0

# all_live <n>: every crate is live. Exit 0 only on a smoke PASS for $TARGET.
all_live() {
    local sline src
    if [ "$smoke_pass_seen" = 1 ]; then
        echo "DRAIN COMPLETE: $1/$TOTAL at $TARGET, post-publish smoke PASS"
        exit 0
    fi
    echo "all $1/$TOTAL crates live at $TARGET, but this drain saw no SMOKE_RESULT PASS for $TARGET; running the post-publish smoke now"
    bash scripts/post_publish_smoke.sh --version "$TARGET" --train-published "$TARGET" 2>&1 \
        | tail -c "$DRAIN_LOG_CAP" >"$smoke_log"
    src=${PIPESTATUS[0]}
    tail -n 20 "$smoke_log"
    sline=$(grep '^SMOKE_RESULT ' "$smoke_log" | tail -1)
    if [ "$src" -eq 0 ] && [ "$sline" = "SMOKE_RESULT PASS version=$TARGET" ]; then
        echo "DRAIN COMPLETE: $1/$TOTAL at $TARGET, post-publish smoke PASS (run by the drain)"
        exit 0
    fi
    case "$sline" in
        "SMOKE_RESULT YANK "*)
            echo "DRAIN FAILED: post-publish smoke (run by the drain, rc=$src): $sline (log: $smoke_log)" >&2
            exit 4 ;;
    esac
    echo "DRAIN NOT VERIFIED: post-publish smoke (run by the drain) rc=$src: ${sline:-no SMOKE_RESULT line} (log: $smoke_log)" >&2
    exit 5
}

for pass in $(seq 1 "$PASSES"); do
    if [ -f "$cascade_log" ]; then mv -f -- "$cascade_log" "$cascade_log.prev"; fi
    n=$(count_pub)
    echo "=== PASS $pass START: $n/$TOTAL at $TARGET ==="
    if [ "$n" -ge "$TOTAL" ]; then all_live "$n"; fi

    # A stale CARGO_REGISTRY_TOKEN 403s every upload; the credentials file wins.
    # The cascade's rc and the smoke result must survive the filter: the capped
    # output goes to the log, the rc is kept with PIPESTATUS (never $? through a
    # pipe), and the summary is read back from the log.
    ( unset CARGO_REGISTRY_TOKEN; bash scripts/cascade-publish.sh ) 2>&1 \
        | tail -c "$DRAIN_LOG_CAP" >"$cascade_log"
    crc=${PIPESTATUS[0]}
    grep -E "PUBLISHED|FATAL|DEFER|already ${TARGET}|TIER|RETRY|SMOKE_RESULT|post-publish smoke" "$cascade_log" | tail -60
    smoke_line=$(grep '^SMOKE_RESULT ' "$cascade_log" | tail -1)
    case "$smoke_line" in
        "SMOKE_RESULT PASS version=$TARGET") smoke_pass_seen=1 ;;
        "") : ;; # no smoke this pass; all_live runs it if every crate turns out live
        "SMOKE_RESULT NOT_MEASURED "*)
            echo "DRAIN NOT VERIFIED: post-publish smoke NOT_MEASURED: $smoke_line (log: $cascade_log)" >&2
            exit 5 ;;
        "SMOKE_RESULT YANK "*)
            echo "DRAIN FAILED: post-publish smoke: $smoke_line (log: $cascade_log)" >&2
            exit 4 ;;
        *)
            echo "DRAIN NOT VERIFIED: unrecognised post-publish smoke line: $smoke_line (log: $cascade_log)" >&2
            exit 5 ;;
    esac
    if [ "$crc" -ne 0 ]; then echo "cascade-publish.sh pass $pass exited rc=$crc (log: $cascade_log)"; fi

    n2=$(count_pub)
    echo "=== PASS $pass END: $n2/$TOTAL ==="
    if [ "$n2" -ge "$TOTAL" ]; then all_live "$n2"; fi
    # No forward progress means a genuine publish error, not a transient —
    # read the log rather than re-running. Never bump the version mid-drain.
    if [ "$n2" -le "$n" ]; then
        echo "STUCK: no progress ($n -> $n2)/$TOTAL after pass $pass" >&2
        exit 2
    fi
done

echo "EXHAUSTED $PASSES passes at $(count_pub)/$TOTAL" >&2
exit 3
