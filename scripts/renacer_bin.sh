#!/usr/bin/env bash
# renacer_bin.sh - resolve the IN-TREE renacer, built from HEAD (TRACE-001 TR-05,
# contract tool-resolution-v1, aprender#4560).
#
#     . scripts/renacer_bin.sh || exit 1     # exports $RENACER
#     "$RENACER" --summary --timing -- ./target/release/examples/foo
#
# WHY. `scripts/capture_golden_traces.sh` ran `cargo install renacer --version
# 0.6.2` from crates.io when no `renacer` was on PATH, and every trace line
# called a bare `renacer`. renacer has lived in this workspace since APR-MONO
# (crates/aprender-profile, `[lib] name = "renacer"`), so the traces came from
# whatever binary PATH happened to hold - a months-old crates.io 0.6.2, or a
# sibling worktree's build - never the code under review. TRACE-001 R-2: in-tree
# tools only, never crates.io, never bare PATH.
#
# THE BINARY'S NAME. The package is `aprender-profile` and its only [[bin]]
# target is `aprender-profile` (src/main.rs); clap names the command `renacer`.
# There is no `target/release/renacer` file, so this resolves
# `<target_dir>/release/aprender-profile` and exports it as $RENACER.
#
# HOW IT MIRRORS scripts/apr_bin.sh:
#   - SOURCED and option-neutral: no file-scope `set`; failure is a RETURN
#     status, never `exit`, so a caller running `set -e` survives a refusal
#     (scripts/check_sourced_libs_option_neutral.sh).
#   - The checkout comes from `git rev-parse --show-toplevel`, never
#     BASH_SOURCE, so sourcing from zsh resolves against the right workspace.
#   - Search roots are the cargo-metadata target_directory and
#     <workspace_root>/target, and nothing else: no PATH, no absolute path.
#   - Attribution: the binary's dep-info (`aprender-profile.d`) must name THIS
#     tree's crates/aprender-profile/src/main.rs. A foreign binary is never
#     returned, fresh or not.
#   - Freshness: `--version` must contain `git rev-parse --short HEAD`
#     (aprender-build-sha stamps it, #4219). A stale binary is refused.
#
# It BUILDS when no own-and-fresh binary exists:
#     cargo build --release -p aprender-profile --bin aprender-profile
# RENACER_BIN_NO_BUILD=1 turns that off (CI steps that must not compile).
#
# Env:
#   RENACER_BIN            use this binary; it must still be executable and fresh
#   RENACER_BIN_NO_BUILD=1 never build; refuse instead
#   RENACER_BIN_TARGET_DIR search root override (self-test seam)
#
# Executed rather than sourced it prints the path; `--self-test` runs the case
# table (target absent -> the sourcing caller survives, among others).

renacer_bin_die() {
    printf 'renacer_bin: %s\n' "$*" >&2
    return 1
}

# Sets RENACER_BIN_TARGET, RENACER_BIN_LOCAL, RENACER_BIN_WS_ROOT, RENACER_BIN_ANCHOR.
renacer_bin_load_meta() {
    local here meta
    RENACER_BIN_TARGET=""
    RENACER_BIN_WS_ROOT=""
    RENACER_BIN_ANCHOR=""
    here=$(git rev-parse --show-toplevel 2>/dev/null) || here=""
    if [ -z "$here" ]; then
        return 1
    fi
    # Every assignment from a command substitution carries `|| x=""`: this file
    # is sourced under `set -e`, and a bare failing one would kill the caller.
    meta=$(cd "$here" && cargo metadata --no-deps --format-version 1 2>/dev/null) || meta=""
    if [ -z "$meta" ]; then
        return 1
    fi
    RENACER_BIN_TARGET=$(printf '%s\n' "$meta" | jq -r '.target_directory // empty' 2>/dev/null) || RENACER_BIN_TARGET=""
    RENACER_BIN_WS_ROOT=$(printf '%s\n' "$meta" | jq -r '.workspace_root // empty' 2>/dev/null) || RENACER_BIN_WS_ROOT=""
    RENACER_BIN_ANCHOR=$(printf '%s\n' "$meta" \
        | jq -r '.packages[] | select(.name == "aprender-profile") | .targets[] | select(.name == "aprender-profile") | select(.kind | index("bin")) | .src_path' 2>/dev/null) || RENACER_BIN_ANCHOR=""
    RENACER_BIN_LOCAL=""
    if [ -n "$RENACER_BIN_WS_ROOT" ] && [ "$RENACER_BIN_WS_ROOT/target" != "$RENACER_BIN_TARGET" ]; then
        RENACER_BIN_LOCAL="$RENACER_BIN_WS_ROOT/target"
    fi
    # The override REPLACES both roots, so a self-test row cannot pick up a
    # real build that happens to sit in <workspace_root>/target.
    if [ -n "${RENACER_BIN_TARGET_DIR:-}" ]; then
        RENACER_BIN_TARGET="$RENACER_BIN_TARGET_DIR"
        RENACER_BIN_LOCAL=""
    fi
    [ -n "$RENACER_BIN_TARGET" ] && [ -n "$RENACER_BIN_ANCHOR" ]
}

# own | foreign | unknown, from the dep-info cargo writes beside the binary.
renacer_bin_origin() {
    local dep="${1}.d"
    if [ ! -f "$dep" ]; then
        printf 'unknown\n'
        return 0
    fi
    if grep -qF -- "$RENACER_BIN_ANCHOR" "$dep" 2>/dev/null; then
        printf 'own\n'
    else
        printf 'foreign\n'
    fi
}

renacer_bin_is_fresh() {
    local bin="$1" head ver
    head=$(git rev-parse --short HEAD 2>/dev/null) || head=""
    if [ -z "$head" ]; then
        return 1
    fi
    ver=$("$bin" --version 2>&1) || ver=""
    case "$ver" in
        *"($head)"*) return 0 ;;
    esac
    return 1
}

# Prints the first own-and-fresh candidate. Fully quoted, no arrays (bash 3.2
# on mini) and no word splitting (zsh does not split unquoted expansions).
renacer_bin_scan() {
    local root cand
    for root in "$RENACER_BIN_TARGET" "$RENACER_BIN_LOCAL"; do
        [ -n "$root" ] || continue
        for cand in "$root/release/aprender-profile" "$root/debug/aprender-profile"; do
            [ -x "$cand" ] || continue
            [ "$(renacer_bin_origin "$cand")" = "own" ] || continue
            renacer_bin_is_fresh "$cand" || continue
            printf '%s\n' "$cand"
            return 0
        done
    done
    return 1
}

renacer_bin_resolve() {
    local found
    if [ -n "${RENACER_BIN:-}" ]; then
        if [ ! -x "$RENACER_BIN" ]; then
            renacer_bin_die "RENACER_BIN=$RENACER_BIN is not an executable file"
            return 1
        fi
        if ! renacer_bin_is_fresh "$RENACER_BIN"; then
            renacer_bin_die "RENACER_BIN=$RENACER_BIN was not built from HEAD ($(git rev-parse --short HEAD 2>/dev/null)); its --version does not carry that sha"
            return 1
        fi
        printf '%s\n' "$RENACER_BIN"
        return 0
    fi
    if ! renacer_bin_load_meta; then
        renacer_bin_die "cannot read cargo metadata for this checkout (need git, cargo, jq, and the aprender-profile bin target)"
        return 1
    fi
    found=$(renacer_bin_scan) || found=""
    if [ -n "$found" ]; then
        printf '%s\n' "$found"
        return 0
    fi
    if [ "${RENACER_BIN_NO_BUILD:-0}" = "1" ]; then
        renacer_bin_die "no aprender-profile built from HEAD under $RENACER_BIN_TARGET and RENACER_BIN_NO_BUILD=1 (build: cargo build --release -p aprender-profile --bin aprender-profile)"
        return 1
    fi
    printf 'renacer_bin: building aprender-profile from HEAD (cargo build --release -p aprender-profile --bin aprender-profile)\n' >&2
    if ! (cd "$RENACER_BIN_WS_ROOT" && cargo build --release -p aprender-profile --bin aprender-profile >&2); then
        renacer_bin_die "cargo build -p aprender-profile failed"
        return 1
    fi
    found=$(renacer_bin_scan) || found=""
    if [ -z "$found" ]; then
        renacer_bin_die "built, but no own-and-fresh aprender-profile appeared under $RENACER_BIN_TARGET (dep-info must name $RENACER_BIN_ANCHOR; --version must carry HEAD)"
        return 1
    fi
    printf '%s\n' "$found"
    return 0
}

# Case table. Each row runs a CALLER shell under `set -euo pipefail` that
# sources this file and then prints `survived rc=<n>`: a refusal that killed
# the caller would print nothing.
renacer_bin_self_test() {
    local self="$1" tmp head pass=0 fail=0 root
    head=$(git rev-parse --short HEAD 2>/dev/null) || head=""
    root=$(git rev-parse --show-toplevel 2>/dev/null) || root=""
    if [ -z "$head" ] || [ -z "$root" ]; then
        printf 'self-test needs a git checkout\n' >&2
        return 2
    fi
    tmp=$(mktemp -d) || tmp=""
    case "$tmp" in /tmp/*) : ;; *) printf 'mktemp gave %s\n' "${tmp:-<empty>}" >&2; return 2 ;; esac
    mkdir -p "$tmp/empty" "$tmp/own/release" "$tmp/foreign/release"
    printf '#!/bin/sh\necho "renacer 0.0.0 (%s)"\n' "$head" > "$tmp/fresh"
    printf '#!/bin/sh\necho "renacer 0.0.0 (0000000)"\n' > "$tmp/stale"
    printf 'not executable\n' > "$tmp/noexec"
    chmod +x "$tmp/fresh" "$tmp/stale"
    cp "$tmp/fresh" "$tmp/own/release/aprender-profile"
    cp "$tmp/fresh" "$tmp/foreign/release/aprender-profile"
    printf '%s: %s/crates/aprender-profile/src/main.rs\n' "$tmp/own/release/aprender-profile" "$root" > "$tmp/own/release/aprender-profile.d"
    printf '%s: /elsewhere/crates/aprender-profile/src/main.rs\n' "$tmp/foreign/release/aprender-profile" > "$tmp/foreign/release/aprender-profile.d"

    _rb_case() { # name want-rc want-path shell env...
        local name="$1" want="$2" wpath="$3" sh="$4" out
        shift 4
        out=$(cd "$root" && env "$@" "$sh" -c 'set -euo pipefail; rc=0; . "$1" || rc=$?; printf "survived rc=%s path=%s opts=%s\n" "$rc" "${RENACER:-}" "$-"' _ "$self" 2>/dev/null) || out="caller died: $out"
        case "$out" in
            *"survived rc=$want path=$wpath opts="*) printf '  ok    %-40s %s\n' "$name" "$sh"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-40s want rc=%s path=%s, got: %s\n' "$name" "$want" "$wpath" "$out"; fail=$((fail + 1)) ;;
        esac
    }
    # THE TR-05 DONE-WHEN ROW: target absent -> the set -e caller survives, rc 1.
    _rb_case target_absent_caller_survives   1 ""                                 bash RENACER_BIN_TARGET_DIR="$tmp/empty" RENACER_BIN_NO_BUILD=1 RENACER_BIN=
    _rb_case own_fresh_resolves              0 "$tmp/own/release/aprender-profile" bash RENACER_BIN_TARGET_DIR="$tmp/own" RENACER_BIN_NO_BUILD=1 RENACER_BIN=
    _rb_case foreign_fresh_is_refused        1 ""                                 bash RENACER_BIN_TARGET_DIR="$tmp/foreign" RENACER_BIN_NO_BUILD=1 RENACER_BIN=
    _rb_case override_fresh_resolves         0 "$tmp/fresh"                       bash RENACER_BIN="$tmp/fresh"
    _rb_case override_stale_is_refused       1 ""                                 bash RENACER_BIN="$tmp/stale"
    _rb_case override_not_executable_refused 1 ""                                 bash RENACER_BIN="$tmp/noexec"
    if command -v zsh >/dev/null 2>&1; then
        _rb_case zsh_target_absent_survives  1 ""                                 zsh RENACER_BIN_TARGET_DIR="$tmp/empty" RENACER_BIN_NO_BUILD=1 RENACER_BIN=
        _rb_case zsh_own_fresh_resolves      0 "$tmp/own/release/aprender-profile" zsh RENACER_BIN_TARGET_DIR="$tmp/own" RENACER_BIN_NO_BUILD=1 RENACER_BIN=
    fi
    # Option neutrality: a caller WITHOUT errexit must not come back with it.
    local opts
    opts=$(cd "$root" && env RENACER_BIN="$tmp/stale" bash -c 'set +e; . "$1"; case "$-" in *e*) echo leaked;; *) echo neutral;; esac' _ "$self" 2>/dev/null) || opts=""
    if [ "$opts" = "neutral" ]; then
        printf '  ok    %-40s bash\n' source_is_option_neutral; pass=$((pass + 1))
    else
        printf '  BROKE %-40s got: %s\n' source_is_option_neutral "$opts"; fail=$((fail + 1))
    fi
    rm -rf "${tmp:?}"
    printf '%s passed, %s broken\n' "$pass" "$fail"
    [ "$fail" = 0 ]
}

# --- executed with --self-test --------------------------------------------
if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]:-}" = "${0}" ] && [ "${1:-}" = "--self-test" ]; then
    renacer_bin_self_test "$0"
    exit $?
fi

# --- resolve ----------------------------------------------------------------
RENACER_BIN_RC=0
RENACER=$(renacer_bin_resolve) || RENACER_BIN_RC=$?
if [ "$RENACER_BIN_RC" -ne 0 ] || [ -z "$RENACER" ]; then
    RENACER=""
    return 1 2>/dev/null || exit 1
fi
export RENACER

# When executed rather than sourced, print the resolved path.
if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]:-}" = "${0}" ]; then
    printf '%s\n' "$RENACER"
fi
