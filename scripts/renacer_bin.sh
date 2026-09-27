#!/usr/bin/env bash
# renacer_bin.sh - resolve the IN-TREE `renacer` and prove it was built from
# this checkout at HEAD (TRACE-001 TR-05, contract tool-resolution-v1).
#
# Sourceable:  . scripts/renacer_bin.sh || exit 1  -> exports $RENACER
# Executable:  bash scripts/renacer_bin.sh         -> prints the path
#              bash scripts/renacer_bin.sh --self-test
#
# WHY THIS EXISTS. renacer is crates/aprender-profile, in this tree. Its binary
# target is `aprender-profile` (src/main.rs; `renacer` is the [lib] name and the
# clap command name), so the file is <target>/release/aprender-profile and
# `--version` prints `renacer ...`. TRACE-001 TR-05 wrote `target/release/renacer`;
# no such file is ever built. scripts/capture_golden_traces.sh installed renacer
# 0.6.2 from crates.io, and the Makefile `profile:` target ran a bare
# `renacer`, which resolves to whatever is first on PATH. Both traced with a
# binary that is not the code under review. The same failure apr_bin.sh exists
# for: a fresh build, a stale execution, a green result.
#
# HOW "BUILT FROM HEAD" IS PROVEN. crates/aprender-profile/build.rs stamps
# APR_GIT_SHA (aprender-build-sha, #4219), so `renacer --version` prints
# `renacer <version> (<short sha>)`. The binary is accepted only if that short
# sha is a prefix of `git rev-parse HEAD` of this checkout. A binary that
# cannot be attributed is rebuilt (RENACER_BIN_BUILD=1, the default) or refused.
#
# OPTION-NEUTRAL. This file is SOURCED; `set` here would change the caller's
# shell (see check_sourced_libs_option_neutral.sh). Failure is a RETURN status,
# never `exit` from a sourced context, so a caller built to tally failures
# survives an absent binary:  . scripts/renacer_bin.sh || skip=1
#
# Knobs:
#   RENACER_BIN        explicit path; still must carry HEAD's sha
#   RENACER_BIN_BUILD  1 (default) = `cargo build --release -p aprender-profile
#                      --bin aprender-profile` when absent or stale; 0 = resolve only
#   CARGO_TARGET_DIR   where the build lands (default <repo>/target)

renacer_bin_root() {
    local here
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd) || return 1
    git -C "$here" rev-parse --show-toplevel 2>/dev/null
}

# Prints the short sha a renacer binary was stamped with, or nothing.
renacer_bin_sha() {
    "$1" --version 2>/dev/null | sed -n 's/^renacer [^ ]* (\([0-9a-f]\{7,\}\))$/\1/p' | head -n 1
}

# 0 when $1 exists, runs, and carries a sha that prefixes HEAD of $2.
renacer_bin_is_head() {
    local bin="$1" root="$2" sha head
    [ -x "$bin" ] || return 1
    sha=$(renacer_bin_sha "$bin")
    [ -n "$sha" ] || return 1
    head=$(git -C "$root" rev-parse HEAD 2>/dev/null) || return 1
    case "$head" in
        "$sha"*) return 0 ;;
    esac
    return 1
}

renacer_bin_resolve() {
    local root target bin
    root=$(renacer_bin_root) || {
        printf 'RENACER UNRESOLVED: not inside a git checkout; cannot prove what renacer was built from\n' >&2
        return 1
    }
    target="${CARGO_TARGET_DIR:-${root}/target}"
    bin="${RENACER_BIN:-${target}/release/aprender-profile}"

    if renacer_bin_is_head "$bin" "$root"; then
        printf '%s\n' "$bin"
        return 0
    fi
    if [ -n "${RENACER_BIN:-}" ] || [ "${RENACER_BIN_BUILD:-1}" != "1" ]; then
        printf 'RENACER UNRESOLVED: %s is absent or not built from HEAD %s (got: %s)\n' \
            "$bin" "$(git -C "$root" rev-parse --short HEAD 2>/dev/null)" \
            "$( [ -x "$bin" ] && "$bin" --version 2>/dev/null | head -n 1 || printf 'no binary')" >&2
        return 1
    fi
    printf 'renacer_bin: building the in-tree renacer at HEAD into %s\n' "$target" >&2
    (cd "$root" && CARGO_TARGET_DIR="$target" cargo build --release -p aprender-profile --bin aprender-profile >&2) || {
        printf 'RENACER UNRESOLVED: cargo build -p aprender-profile --bin aprender-profile failed\n' >&2
        return 1
    }
    if renacer_bin_is_head "$bin" "$root"; then
        printf '%s\n' "$bin"
        return 0
    fi
    printf 'RENACER UNRESOLVED: rebuilt %s but its --version does not carry HEAD (dirty sha override?)\n' "$bin" >&2
    return 1
}

# Target absent -> the resolver returns non-zero, and a caller that sources
# this file keeps running with its own shell options untouched.
renacer_bin_self_test() {
    local root lib st out rc
    root=$(renacer_bin_root) || return 1
    lib="${root}/scripts/renacer_bin.sh"
    # Fixtures live under the target dir and are rewritten in place each run;
    # `absent` is never created, so nothing is deleted.
    st="${CARGO_TARGET_DIR:-${root}/target}/renacer_bin_selftest"
    out=$(CARGO_TARGET_DIR="${st}/absent" RENACER_BIN_BUILD=0 bash -c '
        set -uo pipefail
        before=$-
        . "$1" 2>/dev/null; rc=$?
        [ "$before" = "$-" ] || { echo "OPTIONS-CHANGED $before -> $-"; exit 0; }
        echo "SURVIVED rc=$rc"
    ' _ "$lib")
    rc=$?
    case "$out" in
        "SURVIVED rc=1") ;;
        *) printf 'SELF-TEST FAILED: target absent should return 1 and leave the caller alive, got rc=%s out=%s\n' "$rc" "$out" >&2
           return 1 ;;
    esac
    # A binary whose sha is not HEAD must be refused, not trusted.
    mkdir -p "${st}/foreign/release" || return 1
    printf '#!/bin/sh\necho "renacer 0.0.0 (0000000)"\n' > "${st}/foreign/release/aprender-profile"
    chmod +x "${st}/foreign/release/aprender-profile" || return 1
    out=$(CARGO_TARGET_DIR="${st}/foreign" RENACER_BIN_BUILD=0 bash -c '. "$1" 2>/dev/null; echo "rc=$?"' _ "$lib")
    if [ "$out" != "rc=1" ]; then
        printf 'SELF-TEST FAILED: a renacer stamped 0000000 was accepted (%s)\n' "$out" >&2
        return 1
    fi
    printf 'self-test OK: target absent -> rc=1, caller survives, options unchanged; foreign sha refused\n'
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    renacer_bin_self_test; RENACER_BIN_RC=$?
    return "$RENACER_BIN_RC" 2>/dev/null || exit "$RENACER_BIN_RC"
fi

RENACER=$(renacer_bin_resolve) || { return 1 2>/dev/null || exit 1; }
export RENACER
# Executed rather than sourced: print the path.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    printf '%s\n' "$RENACER"
fi
return 0 2>/dev/null || exit 0
