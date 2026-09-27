#!/usr/bin/env bash
# renacer_bin.sh - resolve the in-tree renacer (crates/aprender-profile) and
# PROVE it was built from THIS TREE at THIS commit. TRACE-001 TR-05 (#4560),
# contract tool-resolution-v1. Mirrors scripts/apr_bin.sh.
#
# Sourceable:  . scripts/renacer_bin.sh || exit 1  -> exports $RENACER
# Executable:  bash scripts/renacer_bin.sh         -> prints the path
#
# WHY. capture_golden_traces.sh ran `cargo install renacer --version 0.6.2` -
# a pre-monorepo crates.io release, not the crate in this tree - and the
# Makefile `profile:` target ran a bare `renacer` from PATH. Either measures
# code nobody here is changing (TRACE-001 R-3: in-tree only, same rule as
# trueno).
#
# THE BINARY IS NOT NAMED `renacer`. crates/aprender-profile has no [[bin]]
# table, so cargo names its src/main.rs binary after the PACKAGE:
# target/release/aprender-profile. The name is read from `cargo metadata`
# rather than typed here, so a later `[[bin]] name = "renacer"` is picked up
# without editing this file (a `renacer` bin target is preferred if present).
#
# PROOF OF ORIGIN, two independent checks (both must hold):
#   1. `--version` prints `<name> <version> (<sha>)` (build.rs, #4219). The
#      sha (aprender-build-sha: `git rev-parse --short HEAD` at build time)
#      must be a PREFIX of the full HEAD. Not compared to a fresh `--short`:
#      git lengthens short shas as the object count grows, so equality with
#      today's `--short` can refuse a binary built from HEAD yesterday.
#   2. The dep-info file `<bin>.d` next to the binary must name THIS
#      workspace's aprender-profile/src/main.rs. Two worktrees at one commit
#      share a sha; only the dep-info tells them apart. A binary with no
#      dep-info is accepted on the sha alone, as apr_bin.sh does.
#
# BUILDS WHEN NEEDED. If no candidate passes, it runs
#   cargo build --release -p aprender-profile --bin <name>
# in the workspace root and checks again. RENACER_BIN_BUILD=0 disables that.
#
# DELIBERATELY NO `set -euo pipefail`. This file is SOURCED; `set` here would
# mutate the caller's shell (see scripts/check_sourced_libs_option_neutral.sh).
# Every failure is a `return` status. Every assignment from a command
# substitution carries `|| x=""`, so a caller running `set -e` survives a
# missing jq or a failing cargo and reaches the refusal instead.
#
# Linux only: aprender-profile is compile_error! elsewhere (TRACE-001 R-4).
#
# Environment:
#   RENACER_BIN_BUILD=0    never build; refuse if nothing fresh exists
#   RENACER_BIN_ROOTS      colon-separated target dirs to search instead of
#                          cargo's target dir and <workspace>/target (selftest)

renacer_bin_die() {
    printf '%s\n' "$*" >&2
    return 1
}

# Workspace root, cargo target dir and the binary's name. One cargo call.
renacer_bin_load_meta() {
    RENACER_BIN_WS_ROOT=""
    RENACER_BIN_TARGET_DIR=""
    RENACER_BIN_NAME=""
    local here meta names
    here=$(git rev-parse --show-toplevel 2>/dev/null) || here=""
    if [ -z "$here" ]; then
        return 1
    fi
    # --manifest-path, not `cd`: the cwd is already inside the tree (git said
    # so), and cargo reads .cargo/config.toml - the target-dir redirect - from
    # the cwd, so the caller's cwd is the one that must stay.
    meta=$(cargo metadata --manifest-path "$here/Cargo.toml" --no-deps --format-version 1 2>/dev/null) || meta=""
    if [ -z "$meta" ]; then
        return 1
    fi
    RENACER_BIN_WS_ROOT=$(printf '%s\n' "$meta" | jq -r '.workspace_root // empty' 2>/dev/null) || RENACER_BIN_WS_ROOT=""
    RENACER_BIN_TARGET_DIR=$(printf '%s\n' "$meta" | jq -r '.target_directory // empty' 2>/dev/null) || RENACER_BIN_TARGET_DIR=""
    names=$(printf '%s\n' "$meta" \
        | jq -r '.packages[] | select(.name == "aprender-profile") | .targets[] | select(.kind | index("bin")) | .name' 2>/dev/null) || names=""
    case "
$names
" in
        *"
renacer
"*) RENACER_BIN_NAME=renacer ;;
        *) RENACER_BIN_NAME=$(printf '%s\n' "$names" | head -n 1) || RENACER_BIN_NAME="" ;;
    esac
    [ -n "$RENACER_BIN_WS_ROOT" ] && [ -n "$RENACER_BIN_NAME" ]
}

# 0 iff the binary's stamped sha is a prefix (>= 7 hex) of HEAD.
renacer_bin_is_fresh() {
    local bin="$1" head stamp
    head=$(git rev-parse HEAD 2>/dev/null) || head=""
    stamp=$("$bin" --version 2>/dev/null | sed -n 's/.*(\([0-9a-f]\{7,40\}\)).*/\1/p' | head -n 1) || stamp=""
    if [ -z "$head" ] || [ -z "$stamp" ]; then
        return 1
    fi
    case "$head" in
        "$stamp"*) return 0 ;;
    esac
    return 1
}

# 0 unless the binary's dep-info names a different workspace.
renacer_bin_is_own() {
    local bin="$1" dep="$1.d"
    if [ ! -f "$dep" ]; then
        return 0
    fi
    grep -qF "$RENACER_BIN_WS_ROOT/crates/aprender-profile/src/main.rs" "$dep"
}

# Print the first candidate that is executable, own and fresh.
renacer_bin_scan() {
    local roots root cand
    roots="${RENACER_BIN_ROOTS:-$RENACER_BIN_TARGET_DIR:$RENACER_BIN_WS_ROOT/target}"
    local IFS=:
    for root in $roots; do
        cand="$root/release/$RENACER_BIN_NAME"
        if [ -x "$cand" ] && renacer_bin_is_own "$cand" && renacer_bin_is_fresh "$cand"; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

renacer_bin_resolve() {
    if ! renacer_bin_load_meta; then
        renacer_bin_die "renacer_bin: cannot read cargo metadata for aprender-profile (not in the aprender workspace, or cargo/jq missing)"
        return 1
    fi
    if renacer_bin_scan; then
        return 0
    fi
    if [ "${RENACER_BIN_BUILD:-1}" = "0" ]; then
        renacer_bin_die "renacer_bin: no $RENACER_BIN_NAME built from HEAD in this tree, and RENACER_BIN_BUILD=0"
        return 1
    fi
    printf 'renacer_bin: building %s from HEAD (cargo build --release -p aprender-profile)\n' "$RENACER_BIN_NAME" >&2
    if ! cargo build --manifest-path "$RENACER_BIN_WS_ROOT/Cargo.toml" --release -p aprender-profile --bin "$RENACER_BIN_NAME" >&2; then
        renacer_bin_die "renacer_bin: cargo build -p aprender-profile failed"
        return 1
    fi
    if renacer_bin_scan; then
        return 0
    fi
    renacer_bin_die "renacer_bin: built $RENACER_BIN_NAME but no candidate proves it came from HEAD in this tree"
    return 1
}

RENACER_BIN_RC=0
RENACER=$(renacer_bin_resolve) || RENACER_BIN_RC=$?
if [ "$RENACER_BIN_RC" -ne 0 ]; then
    RENACER=""
    return 1 2>/dev/null || exit 1
fi
export RENACER

# When executed rather than sourced, print the resolved path.
if [ "${BASH_SOURCE[0]:-}" = "${0}" ]; then
    printf '%s\n' "$RENACER"
fi
