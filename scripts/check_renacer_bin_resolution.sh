#!/usr/bin/env bash
# check_renacer_bin_resolution.sh - selftest for scripts/renacer_bin.sh
# (TRACE-001 TR-05, #4560, contract tool-resolution-v1).
#
# Every REFUSE case is paired with an ACCEPT control that differs in exactly
# the property under test, so a resolver that refuses everything (or accepts
# everything) fails. Binaries are SYNTHESIZED in a temp dir and searched via
# RENACER_BIN_ROOTS with RENACER_BIN_BUILD=0: the test never builds, never
# depends on what happens to be in target/, and runs in seconds.
#
#   R1  target absent            -> refused, and a `set -e` caller SURVIVES
#   R2  stale sha                -> refused      A2 HEAD sha      -> accepted
#   R3  foreign dep-info         -> refused      A3 own dep-info  -> accepted
#   R4  no sha in --version      -> refused
#   A4  executed (not sourced)   -> prints the path
#   N1  sourcing leaves the caller's shell options unchanged
#   M1  bin name comes from cargo metadata, not a literal
set -euo pipefail
cd "$(dirname "$0")/.."

LIB=scripts/renacer_bin.sh
WS=$(git rev-parse --show-toplevel)
HEAD=$(git rev-parse HEAD)
NAME=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
    | jq -r '.packages[] | select(.name == "aprender-profile") | .targets[] | select(.kind | index("bin")) | .name' \
    | head -n 1)

TMP=$(mktemp -d)
cleanup() { rm -rf "${TMP:?}"; }
trap cleanup EXIT

FAIL=0
pass() { printf 'PASS %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; FAIL=1; }

# mkbin <root> <version-line> [dep-info-source]
mkbin() {
    mkdir -p "$1/release"
    printf '#!/bin/sh\necho "%s"\n' "$2" > "$1/release/$NAME"
    chmod +x "$1/release/$NAME"
    if [ -n "${3:-}" ]; then
        printf '%s: %s\n' "$1/release/$NAME" "$3" > "$1/release/$NAME.d"
    fi
}

# resolve <root> -> prints "rc=<n> RENACER=<path>"; the caller runs set -e.
resolve() {
    RENACER_BIN_BUILD=0 RENACER_BIN_ROOTS="$1" bash -c '
        set -euo pipefail
        rc=0
        . "$1" 2>/dev/null || rc=$?
        printf "rc=%s RENACER=%s\n" "$rc" "${RENACER:-}"
    ' check "$LIB"
}

# M1
if [ "$NAME" = "renacer" ] || [ "$NAME" = "aprender-profile" ]; then
    pass "M1 bin name from cargo metadata: $NAME"
else
    fail "M1 unexpected aprender-profile bin name: '$NAME'"
fi
if grep -qE '^[^#]*release/(renacer|aprender-profile)' "$LIB"; then
    fail "M1 $LIB hard-codes the binary name"
else
    pass "M1 $LIB does not hard-code the binary name"
fi

# R1 - nothing built: the caller must survive to report the refusal.
mkdir -p "$TMP/empty"
out=$(resolve "$TMP/empty") || out="caller-died"
case "$out" in
    "rc=1 RENACER=") pass "R1 target absent -> refused, set -e caller survived" ;;
    *) fail "R1 target absent: $out" ;;
esac

# R2 / A2 - stale vs fresh sha.
mkbin "$TMP/stale" "$NAME 0.0.0 (deadbeef00)"
out=$(resolve "$TMP/stale") || out="caller-died"
case "$out" in
    "rc=1 RENACER=") pass "R2 stale sha -> refused" ;;
    *) fail "R2 stale sha: $out" ;;
esac
mkbin "$TMP/fresh" "$NAME 0.0.0 (${HEAD:0:10})"
out=$(resolve "$TMP/fresh") || out="caller-died"
case "$out" in
    "rc=0 RENACER=$TMP/fresh/release/$NAME") pass "A2 HEAD sha -> accepted" ;;
    *) fail "A2 HEAD sha: $out" ;;
esac

# R3 / A3 - same fresh sha, dep-info from another worktree vs this one.
mkbin "$TMP/foreign" "$NAME 0.0.0 (${HEAD:0:10})" "/elsewhere/crates/aprender-profile/src/main.rs"
out=$(resolve "$TMP/foreign") || out="caller-died"
case "$out" in
    "rc=1 RENACER=") pass "R3 foreign dep-info -> refused" ;;
    *) fail "R3 foreign dep-info: $out" ;;
esac
mkbin "$TMP/own" "$NAME 0.0.0 (${HEAD:0:10})" "$WS/crates/aprender-profile/src/main.rs"
out=$(resolve "$TMP/own") || out="caller-died"
case "$out" in
    "rc=0 RENACER=$TMP/own/release/$NAME") pass "A3 own dep-info -> accepted" ;;
    *) fail "A3 own dep-info: $out" ;;
esac

# R4 - a binary that stamps no sha proves nothing.
mkbin "$TMP/nosha" "$NAME 0.0.0"
out=$(resolve "$TMP/nosha") || out="caller-died"
case "$out" in
    "rc=1 RENACER=") pass "R4 no sha stamp -> refused" ;;
    *) fail "R4 no sha stamp: $out" ;;
esac

# A4 - executed mode prints the path.
out=$(RENACER_BIN_BUILD=0 RENACER_BIN_ROOTS="$TMP/fresh" bash "$LIB" 2>/dev/null) || out="rc=$?"
if [ "$out" = "$TMP/fresh/release/$NAME" ]; then
    pass "A4 executed -> prints path"
else
    fail "A4 executed: $out"
fi

# N1 - sourcing (success AND refusal) leaves the caller's options alone.
for root in "$TMP/fresh" "$TMP/empty"; do
    out=$(RENACER_BIN_BUILD=0 RENACER_BIN_ROOTS="$root" bash -c '
        before="$-|$(set -o | tr -s " " | sort | tr "\n" ,)"
        . "$1" 2>/dev/null || true
        after="$-|$(set -o | tr -s " " | sort | tr "\n" ,)"
        [ "$before" = "$after" ] && echo same || echo changed
    ' check "$LIB") || out="caller-died"
    if [ "$out" = "same" ]; then
        pass "N1 options unchanged after sourcing (${root##*/})"
    else
        fail "N1 options changed after sourcing (${root##*/}): $out"
    fi
done

if [ "$FAIL" -ne 0 ]; then
    printf 'check_renacer_bin_resolution: FAILED\n'
    exit 1
fi
printf 'check_renacer_bin_resolution: all cases pass\n'
