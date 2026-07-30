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

set -euo pipefail

apr_bin_die() {
    printf '%s\n' "$*" >&2
    return 1
}

# Resolve the intended binary, most explicit first.
apr_bin_resolve() {
    if [ -n "${APR_BIN:-}" ]; then
        printf '%s\n' "$APR_BIN"
        return 0
    fi
    local cargo_home="${CARGO_HOME:-$HOME/.cargo}"
    if [ -x "$cargo_home/bin/apr" ]; then
        printf '%s\n' "$cargo_home/bin/apr"
        return 0
    fi
    command -v apr 2>/dev/null || return 1
}

# Assert the binary's embedded SHA matches HEAD. No-op outside a git checkout
# (an end user running a released apr has no HEAD to compare against).
apr_bin_assert_fresh() {
    local bin="$1"
    local reported head shadows

    reported=$("$bin" --version 2>&1 || true)

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

APR=$(apr_bin_resolve) || apr_bin_die "apr_bin: no apr binary found (build one: cargo install --path crates/apr-cli --force)"
apr_bin_assert_fresh "$APR"
export APR

# When executed rather than sourced, print the resolved path.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    printf '%s\n' "$APR"
fi
