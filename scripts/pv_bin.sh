#!/usr/bin/env bash
# pv_bin.sh - resolve the `pv` binary and PROVE it is the one THIS checkout builds.
#
# Sourceable:  . scripts/pv_bin.sh   -> exports $PV
# Executable:  bash scripts/pv_bin.sh -> prints the path, exits non-zero if stale
#
# WHY THIS EXISTS. `pv` on PATH is a `cargo install`ed copy that nobody
# reinstalls, and the in-tree crate moves every week. Measured on 2026-08-20 in
# a worktree at origin/main (773a39da1):
#
#   /home/noah/.cargo/bin/pv          pv 0.49.0   installed 2026-06-13
#   crates/aprender-contracts-cli     pv 0.63.0   HEAD
#
# They agree on `validate` and `lint`. They DISAGREE on the gate that matters:
#
#   $ /home/noah/.cargo/bin/pv lint contracts --strict-test-binding --format json --no-cache
#   {"type":"verify","total_refs":253,"existing":202,"missing":51}
#   $ cargo run -q -p aprender-contracts-cli --bin pv -- lint contracts \
#         --strict-test-binding --format json --no-cache
#   {"type":"verify","total_refs":371,"existing":344,"missing":27}
#
# 118 test references invisible, 24 phantom "missing" entries. A developer -- or
# a release-certifying script -- running bare `pv` gets a materially different
# verdict from CI, on the same tree, in the same second. scripts/dogfood_surfaces.sh
# certified releases through exactly that PATH binary until this file landed.
#
# HOW THIS DIFFERS FROM scripts/apr_bin.sh, AND WHY
# -------------------------------------------------
# apr_bin.sh SEARCHES for a binary and then proves freshness by comparing the
# git SHA that crates/apr-cli/build.rs embeds into `apr --version`.
#
# aprender-contracts-cli has NO build.rs, so `pv --version` carries no SHA --
# only `pv 0.63.0`, the workspace version. A version compare is therefore a
# necessary control but NOT a sufficient one, and this was measured rather than
# assumed: during the 2026-08-20 audit two `pv` binaries, BOTH self-reporting
# `pv 0.63.0`, BOTH nominally built from 773a39da, returned 253/51 and 371/27 on
# an identical tree (md5 9c7fac4a... vs 016fc6f6...). Every checkout and every
# worktree here resolves to ONE shared cargo target dir
# (/mnt/nvme-raid0/targets/aprender), so a `pv` sitting in target/debug may have
# been produced by another branch's build minutes ago.
#
# So this file does not SEARCH at all. It asks cargo to BUILD, and names cargo's
# own output. The build is the freshness proof; the version compare is the
# non-vacuity control that catches a PV_BIN override pointing at 0.49.0. That is
# the same conclusion scripts/check_contract_test_binding.sh:95-99 reached
# independently: "Asking cargo to build-and-run is the only resolution that
# cannot pick up someone else's artifact."
#
# RESIDUAL GAP, STATED RATHER THAN HIDDEN. Between our `cargo build` and a
# caller's last use of "$PV", a concurrent build from another worktree can
# replace the file. `pv_bin_assert_unchanged` closes it for callers that care:
# it re-hashes and fails if the artifact moved under them. Long sweeps should
# call it at the end; a single invocation does not need it.
#
# DELIBERATELY NO `set -euo pipefail` AT FILE SCOPE. This file is SOURCED, and
# `set` in a sourced file mutates the CALLER's shell. apr_bin.sh shipped that
# bug: qwen-story.sh chose `set -uo pipefail` WITHOUT `-e` on purpose so it could
# run every beat and tally failures, and the leaked errexit killed the nightly
# six lines in. scripts/dogfood_surfaces.sh -- the first consumer of this file --
# makes the identical choice at line 66 for the identical reason. A sourceable
# library must be option-neutral and fail by RETURN STATUS:
#     . scripts/pv_bin.sh || exit 1
# scripts/check_sourced_libs_option_neutral.sh enforces this.

pv_bin_die() {
    printf '%s\n' "$*" >&2
    return 1
}

# Locate the checkout via git, NOT via the script's own path. `${BASH_SOURCE[0]}`
# is a BASH-ONLY variable; sourcing from zsh (the interactive shell on this dev
# box) leaves it empty, `dirname ""` gives `.`, and `cargo metadata` then reports
# a DIFFERENT workspace's target dir. apr_bin.sh had to be fixed for exactly
# this. `git rev-parse --show-toplevel` is correct under worktrees too.
pv_bin_root() {
    local here
    here=$(git rev-parse --show-toplevel 2>/dev/null) || here=$(pwd)
    [ -n "$here" ] || here=$(pwd)
    printf '%s\n' "$here"
}

# One `cargo metadata` call; both facts come out of it. Splitting this into two
# calls doubles a ~150ms cost on every sourcing script for no benefit.
pv_bin_meta() {
    local root
    root=$(pv_bin_root)
    (cd "$root" && cargo metadata --no-deps --format-version 1 2>/dev/null)
}

# Build, then name cargo's output. Never search PATH, $CARGO_HOME/bin, or an
# absolute path: every strategy that SEARCHES eventually finds the wrong one,
# and here the wrong one is 68 days old.
#
# Sets the globals PV and PV_CRATE_VERSION rather than PRINTING the path, and
# that is not a style choice. `PV=$(pv_bin_resolve)` runs the function inside a
# command-substitution SUBSHELL, so `PV_CRATE_VERSION` assigned in there dies
# with the subshell; the freshness assertion then compares against an empty
# string. Under a caller running `set -u` -- which scripts/dogfood-book.sh:12
# does -- that is not even a silent pass, it is `PV_CRATE_VERSION: unbound
# variable` and the sourcing script exits 1 with no explanation. Caught by
# running the resolver under `set -u` instead of only under the shell that
# happened to be handy.
pv_bin_resolve() {
    local root meta td profile flag out

    root=$(pv_bin_root) || return 1
    meta=$(pv_bin_meta) || return 1
    [ -n "$meta" ] || return 1

    PV_CRATE_VERSION=$(printf '%s' "$meta" \
        | jq -r '.packages[] | select(.name=="aprender-contracts-cli") | .version' 2>/dev/null)
    td=$(printf '%s' "$meta" | jq -r '.target_directory // empty' 2>/dev/null)
    [ -n "$td" ] || return 1
    [ -n "$PV_CRATE_VERSION" ] || return 1

    # Explicit override is the ONLY escape hatch - needed for A/B work such as
    # comparing a released pv against HEAD, which is how the 0.49.0-vs-0.63.0
    # divergence above was measured in the first place. It is still version-
    # checked below, so it cannot be used to smuggle a stale binary past a gate
    # that did not ask for one.
    if [ -n "${PV_BIN:-}" ]; then
        PV="$PV_BIN"
        return 0
    fi

    profile="${PV_PROFILE:-debug}"
    case "$profile" in
        debug)   flag='' ;;
        release) flag='--release' ;;
        *) pv_bin_die "pv_bin: PV_PROFILE must be debug or release, got '$profile'"; return 1 ;;
    esac

    # THE FRESHNESS PROOF. Not a heuristic: cargo's fingerprinting is what makes
    # the artifact match the source. Output is captured so a build failure prints
    # something actionable instead of interleaving with the caller's receipt.
    # `rc` is captured on the SAME line as the assignment. Reading `$?` after an
    # `if ... fi` reads the status the block happened to leave behind, which is
    # the "never read $? through a pipe" defect wearing different clothes.
    local rc=0
    if [ -n "$flag" ]; then
        out=$( (cd "$root" && cargo build -q "$flag" -p aprender-contracts-cli --bin pv) 2>&1 ); rc=$?
    else
        out=$( (cd "$root" && cargo build -q -p aprender-contracts-cli --bin pv) 2>&1 ); rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        {
            printf 'pv_bin: cargo build of aprender-contracts-cli failed:\n'
            printf '%s\n' "$out"
        } >&2
        return 1
    fi

    [ -x "$td/$profile/pv" ] || {
        pv_bin_die "pv_bin: cargo build succeeded but $td/$profile/pv is not executable"
        return 1
    }
    PV="$td/$profile/pv"
    return 0
}

# Non-vacuity control. If this can never fail it is not a check, so it is aimed
# at the one input that DOES fail it: a PV_BIN override naming ~/.cargo/bin/pv.
pv_bin_assert_fresh() {
    local bin="$1" reported shadows
    reported=$("$bin" --version 2>&1 || true)

    case "$reported" in
        "pv $PV_CRATE_VERSION"*) return 0 ;;
        *) ;;
    esac

    # `type -aP` lists every PATH match in resolution order. (`command -v -a` is
    # not valid bash; -a is not a `command` option.) Naming the shadows is the
    # point - the defect is invisible until you see 0.49.0 sitting first.
    shadows=$(type -aP pv 2>/dev/null | awk '!seen[$0]++' || true)
    {
        printf 'STALE pv BINARY\n'
        printf '  resolved : %s\n' "$bin"
        printf '  reports  : %s\n' "$reported"
        printf '  expected : pv %s   (aprender-contracts-cli in this checkout)\n' "$PV_CRATE_VERSION"
        printf '  A pv from another version answers strict-test-binding differently:\n'
        printf '  0.49.0 said 253 refs / 51 missing where HEAD said 371 / 27, on the\n'
        printf '  same tree. Anything it validates says nothing about this checkout.\n'
        if [ -n "$shadows" ]; then
            printf '  every pv on PATH (first wins):\n'
            printf '%s\n' "$shadows" | while IFS= read -r p; do
                [ -n "$p" ] || continue
                printf '    %-40s %s\n' "$p" "$("$p" --version 2>&1 | head -1)"
            done
        fi
        printf '  fix: unset PV_BIN and re-source scripts/pv_bin.sh, which builds\n'
        printf '       the binary this checkout defines.\n'
    } >&2
    return 1
}

# Closes the shared-target-dir race for callers that make many pv calls: proves
# nobody rebuilt over our artifact mid-run. Returns non-zero if it moved.
pv_bin_assert_unchanged() {
    local now
    [ -n "${PV:-}" ] || { pv_bin_die "pv_bin: \$PV is not set"; return 1; }
    [ -n "${PV_SHA256:-}" ] || { pv_bin_die "pv_bin: \$PV_SHA256 is not set"; return 1; }
    now=$(sha256sum "$PV" 2>/dev/null | cut -d' ' -f1)
    [ "$now" = "$PV_SHA256" ] && return 0
    {
        printf 'pv BINARY CHANGED MID-RUN\n'
        printf '  path   : %s\n' "$PV"
        printf '  was    : %s\n' "$PV_SHA256"
        printf '  now    : %s\n' "$now"
        printf '  Every checkout here shares one cargo target dir, so another\n'
        printf '  branch built over it. Measurements taken after this point may\n'
        printf '  describe a different tree. Re-run.\n'
    } >&2
    return 1
}

# Fail-closed without errexit. At the top level of a SOURCED file `return N` ends
# the source with that status, so `. pv_bin.sh || exit 1` behaves exactly as it
# would under `set -e`. When EXECUTED instead, top-level `return` is an error and
# the `||` falls through to `exit`. One line covers both entry points.
PV=''
PV_CRATE_VERSION=''
if ! pv_bin_resolve; then
    pv_bin_die "pv_bin: could not build pv (try: cargo build -p aprender-contracts-cli --bin pv)"
    return 1 2>/dev/null || exit 1
fi

if ! pv_bin_assert_fresh "$PV"; then
    return 1 2>/dev/null || exit 1
fi

PV_SHA256=$(sha256sum "$PV" 2>/dev/null | cut -d' ' -f1)

export PV PV_SHA256 PV_CRATE_VERSION

# When executed rather than sourced, print the resolved path.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    printf '%s\n' "$PV"
fi
