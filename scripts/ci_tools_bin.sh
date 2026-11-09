#!/usr/bin/env bash
# ci_tools_bin.sh - resolve the aprender-ci-tools binary (crates/aprender-ci-tools),
# the Rust home of the scripts' former Python helpers.
#
# Sourceable:  . scripts/ci_tools_bin.sh || exit 1   -> exports $CI_TOOLS_BIN
#
# A $CI_TOOLS_BIN the caller set is used as given, and one that is not an executable
# file is a refusal, never a silent rebuild: the caller asked for THAT binary. Unset,
# the binary is built from this tree (`cargo build -q -p aprender-ci-tools`) and
# located through `cargo metadata`, since the target dir is redirected per checkout.
#
# Option-neutral: sourced, so no `set` at file scope (it would change the caller's
# shell). Failure is the return status, as in scripts/apr_bin.sh.

_ci_tools_bin_resolve() {
    local root tdir
    if [ -n "${CI_TOOLS_BIN:-}" ]; then
        if [ -f "$CI_TOOLS_BIN" ] && [ -x "$CI_TOOLS_BIN" ]; then
            return 0
        fi
        printf 'ci_tools_bin: CI_TOOLS_BIN=%s is not an executable file\n' "$CI_TOOLS_BIN" >&2
        return 1
    fi
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    if ! cargo build -q --locked --manifest-path "$root/Cargo.toml" -p aprender-ci-tools; then
        printf 'ci_tools_bin: cargo build -p aprender-ci-tools failed\n' >&2
        return 1
    fi
    tdir="$(cargo metadata --no-deps --format-version 1 --manifest-path "$root/Cargo.toml" |
        grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)" || return 1
    if [ ! -x "$tdir/debug/aprender-ci-tools" ]; then
        printf 'ci_tools_bin: no binary at %s/debug/aprender-ci-tools after the build\n' "$tdir" >&2
        return 1
    fi
    CI_TOOLS_BIN="$tdir/debug/aprender-ci-tools"
    export CI_TOOLS_BIN
}

_ci_tools_bin_resolve
