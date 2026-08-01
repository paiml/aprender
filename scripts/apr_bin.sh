#!/usr/bin/env bash
# apr_bin.sh - resolve the `apr` binary and PROVE it was built from this commit.
#
# Sourceable:  . scripts/apr_bin.sh   -> exports $APR
# Executable:  bash scripts/apr_bin.sh -> prints the path, exits non-zero if stale
#
# WHY THIS EXISTS. qwen-story-daily ran `cargo install --path crates/apr-cli
# --force` (writing ~/.cargo/bin/apr), then invoked bare `apr` - which resolved
# to a 24-day-old 0.60.0 binary in ~/.local/bin, earlier on PATH:
#
#   /home/noah/.local/bin/apr   0.60.0             2026-07-06   <- won
#   /home/noah/.cargo/bin/apr   0.61.0 (e514cc5ed) 2026-07-30
#
# Fresh install, stale execution, green result. Every beat in that story
# validated July 6 code, including a gate merged the previous day.
#
# The check is exact rather than heuristic because crates/apr-cli/build.rs
# already embeds the build-time git SHA (contract apr-version-traceability-v1,
# F-VERSION-001..004): `apr --version` prints e.g. `apr 0.61.0 (e514cc5ed)`.
# So "was this binary built from HEAD?" is a string comparison, not a guess.
#
# DELIBERATELY NO `set -euo pipefail` AT FILE SCOPE. This file is SOURCED, and
# `set` in a sourced file mutates the CALLER's shell. The first version of this
# script did `set -euo pipefail` here; qwen-story.sh sources it and had chosen
# `set -uo pipefail` WITHOUT `-e` on purpose - the whole script is built around
# running every beat and counting failures (emit_fail/FAILED_BEATS). Turning
# errexit on under it meant the first non-zero command anywhere killed the run:
# the nightly story died after SIX LINES, inside Beat 1's advisory pmat hunt,
# and reported nothing about why. A sourceable library must be option-neutral.
#
# Fail-closed is preserved explicitly instead: the bottom of this file returns
# non-zero when the binary is stale, so callers use
#     . scripts/apr_bin.sh || exit 1
# which fails the same way errexit would, without seizing the caller's shell.

apr_bin_die() {
    printf '%s\n' "$*" >&2
    return 1
}

# Resolve the binary THIS CHECKOUT builds. Ask cargo; never guess.
#
# This deliberately does NOT consult PATH, $CARGO_HOME/bin, or any absolute
# path. Four `apr` binaries were found coexisting on one machine - 0.60.0,
# 0.61.0, 0.62.0 and a second 0.60.0 with no embedded SHA at all - and every
# resolution strategy that *searches* for a binary will eventually find the
# wrong one. Detection was already in place and still lost 24 days to a stale
# nightly, so the fix is to stop searching.
#
# `cargo metadata` reports the real target directory under every checkout shape,
# which matters because `.cargo/config.toml` redirects it and is GITIGNORED
# (.gitignore:55) while its siblings `.cargo/audit.toml` and `.cargo/mutants.toml`
# are tracked. The directory looks version-controlled; the file that moves every
# build output is not. Measured:
#   main checkout  -> /mnt/nvme-raid0/coverage/aprender   (redirect applies)
#   fresh worktree -> <worktree>/target                   (no config, cargo default)
# A hardcoded absolute path is therefore right in exactly one of those and
# silently wrong in the other - which is how a release smoke-test came to read a
# five-hour-old binary and report a meaningless pass.
apr_bin_target_dir() {
    local here
    here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    (cd "$here" && cargo metadata --no-deps --format-version 1 2>/dev/null) \
        | jq -r '.target_directory // empty' 2>/dev/null
}

apr_bin_resolve() {
    # Explicit override is the ONLY escape hatch - needed for A/B work such as
    # comparing a released binary against HEAD. It is still freshness-checked
    # below, so it cannot be used to smuggle a stale binary past the gate.
    if [ -n "${APR_BIN:-}" ]; then
        printf '%s\n' "$APR_BIN"
        return 0
    fi
    local td
    td=$(apr_bin_target_dir)
    [ -n "$td" ] || return 1
    if [ -x "$td/release/apr" ]; then
        printf '%s\n' "$td/release/apr"
        return 0
    fi
    if [ -x "$td/debug/apr" ]; then
        printf '%s\n' "$td/debug/apr"
        return 0
    fi
    # Last resort: the `cargo install` destination. `cargo install` builds in a
    # temp dir and copies only the finished binary here, so it leaves nothing in
    # the target dir above - qwen-story-daily installs exactly this way. Kept
    # LAST so a checkout that has built its own binary always tests that one,
    # and it is still freshness-checked like every other candidate, so this is a
    # fallback in resolution order only, never a way around the gate.
    local cargo_home="${CARGO_HOME:-$HOME/.cargo}"
    if [ -x "$cargo_home/bin/apr" ]; then
        printf '%s\n' "$cargo_home/bin/apr"
        return 0
    fi
    return 1
}

# Assert the binary's embedded SHA matches HEAD. No-op outside a git checkout
# (an end user running a released apr has no HEAD to compare against).
apr_bin_assert_fresh() {
    local bin="$1"
    local reported head shadows

    reported=$("$bin" --version 2>&1 || true)

    # FAIL CLOSED outside a git checkout when strict. The old behaviour returned
    # 0 here ("freshness not asserted"), which meant any binary passed the guard
    # as long as you ran it from the wrong directory - a fail-OPEN hole in a
    # script whose entire job is to refuse unproven binaries. Release and
    # dogfood surfaces set APR_BIN_STRICT=1 so "cannot prove" means "refuse".
    if [ "${APR_BIN_STRICT:-0}" = "1" ] && ! git rev-parse --short HEAD >/dev/null 2>&1; then
        printf 'apr_bin: STRICT mode and this is not a git checkout - cannot prove %s\n' "$bin" >&2
        printf '         was built from the code under test, so it is refused.\n' >&2
        return 1
    fi

    if ! git rev-parse --short HEAD >/dev/null 2>&1; then
        printf 'apr_bin: %s (%s) - not a git checkout, freshness not asserted\n' \
            "$bin" "$reported" >&2
        return 0
    fi
    head=$(git rev-parse --short HEAD)

    case "$reported" in
        *"$head"*) return 0 ;;
        *) ;;
    esac

    # Stale. Name every `apr` on PATH so the shadowing is obvious rather than
    # something the reader has to go discover.
    # `command -v -a` is not valid bash (-a is not a `command` option); use
    # `type -aP`, which lists every PATH match in resolution order.
    shadows=$(type -aP apr 2>/dev/null | awk '!seen[$0]++' || true)
    {
        printf 'STALE apr BINARY\n'
        printf '  resolved : %s\n' "$bin"
        printf '  reports  : %s\n' "$reported"
        printf '  HEAD     : %s\n' "$head"
        printf '  The binary was NOT built from this commit, so anything it\n'
        printf '  validates says nothing about this checkout.\n'
        if [ -n "$shadows" ]; then
            printf '  every apr on PATH (first wins):\n'
            printf '%s\n' "$shadows" | while IFS= read -r p; do
                [ -n "$p" ] || continue
                printf '    %-40s %s\n' "$p" "$("$p" --version 2>&1 | head -1)"
            done
        fi
        printf '  fix: cargo install --path crates/apr-cli --force, then put\n'
        printf '       "$CARGO_HOME/bin" (or ~/.cargo/bin) FIRST on PATH,\n'
        printf '       or set APR_BIN to the binary you mean.\n'
    } >&2
    return 1
}

# Fail-closed without errexit. At the top level of a SOURCED file `return N`
# ends the source with that status, so `. apr_bin.sh || exit 1` behaves exactly
# as it did under `set -e`. When this file is EXECUTED instead, top-level
# `return` is an error, and the `||` falls through to `exit`. One line covers
# both entry points.
if ! APR=$(apr_bin_resolve); then
    apr_bin_die "apr_bin: no apr binary found (build one: cargo install --path crates/apr-cli --force)"
    return 1 2>/dev/null || exit 1
fi

if ! apr_bin_assert_fresh "$APR"; then
    return 1 2>/dev/null || exit 1
fi

export APR

# When executed rather than sourced, print the resolved path.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    printf '%s\n' "$APR"
fi
