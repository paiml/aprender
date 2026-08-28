#!/usr/bin/env bash
# apr_bin.sh - resolve the `apr` binary and PROVE it was built from THIS TREE
# at THIS commit.
#
# Sourceable:  . scripts/apr_bin.sh   -> exports $APR
# Executable:  bash scripts/apr_bin.sh -> prints the path, exits non-zero if the
#                                         binary cannot be attributed to this
#                                         tree at this commit
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
# non-zero when the binary cannot be proven, so callers use
#     . scripts/apr_bin.sh || exit 1
# which fails the same way errexit would, without seizing the caller's shell.
#
# ---------------------------------------------------------------------------
# THE COMMIT SHA IS NOT ENOUGH: TREE ATTRIBUTION (#2739)
# ---------------------------------------------------------------------------
# Everything above proves WHICH COMMIT a binary was built from. It says nothing
# about WHICH TREE. With several agents each holding a git worktree of this
# repo, that gap is load-bearing, because the worktrees share one target
# directory. Measured on this box, from worktree `s2-aprbin`:
#
#   git rev-parse --show-toplevel   .../scratchpad/s2-aprbin
#   cargo metadata target_directory /mnt/nvme-raid0/targets/aprender
#   /mnt/.../release/apr            apr 0.64.0 (de0e3e182)
#   /mnt/.../release/apr.d          ... /scratchpad/p3-warmup/src/bin/apr.rs
#
# The binary sitting in "this checkout's" target dir was built from a DIFFERENT
# worktree. The redirect is not in any tracked file: the dev shell defines a
# `cargo` function that exports
#     CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/$(basename "$(git remote origin url)")
# so every worktree of aprender - all 40+ of them - builds into the SAME
# directory, and whichever tree built last owns <target>/release/apr.
#
# It gets worse: that function only exists in the interactive shell. A build
# launched from a `#!/usr/bin/env bash` script gets the real cargo and lands in
# <worktree>/target instead. So THE SAME TREE builds to two different places
# depending on which shell launched the build, and `cargo metadata` reports
# whichever one applies to the shell asking - not necessarily the one that
# produced the artifact. Both directions were observed live:
#
#   sourced from zsh  -> /mnt/nvme-raid0/targets/aprender/release/apr (p3-warmup's)
#   sourced from bash -> ~/.cargo/bin/apr, installed from the CI runner's
#                        checkout, per ~/.cargo/.crates2.json:
#                        "apr-cli 0.64.0 (path+file:///home/noah/actions-runner-lambda/
#                         _work/aprender/aprender/crates/apr-cli)" = ["apr", ...]
#
# In both cases the old code reported STALE. STALE is the wrong diagnosis and
# sends the reader to `cargo install`, which does not fix it - there is no
# stale binary, there is SOMEONE ELSE'S binary. An agent working #2730 was
# handed a sibling worktree's apr built WITHOUT --features cuda while working a
# CUDA-only path, and it was caught only because the CUDA error text was wrong
# for a CUDA build. Nothing in this protocol would have caught it.
#
# HOW A BINARY IS ATTRIBUTED TO A TREE. Cargo writes a dep-info file next to
# every binary it links: <target>/<profile>/apr.d, a make-style rule whose
# right-hand side lists the ABSOLUTE PATH of every source that went into it.
# That is cargo's own record of which tree it compiled, it is written in the
# same step that writes the binary, and it is plain text on every host. The
# anchor compared against it is the `src_path` cargo metadata reports for the
# bin target named `apr` in THIS workspace, so both sides of the comparison are
# cargo's own path normalisation and cannot disagree about symlinks.
#
# `cargo install` leaves no dep-info - it builds in a temp dir and copies only
# the finished binary - so that candidate is attributed from
# $CARGO_HOME/.crates2.json, which records the `path+file://` the bin was
# installed FROM. Same question, the only record cargo keeps of the answer.
#
# WHY NOT JUST ASK CARGO. `cargo build --bin apr --message-format=json` prints
# the executable path authoritatively, and is the obviously right answer - for
# a build script. It is wrong here: this file is SOURCED by every diagnostic in
# the repo, and a resolver that may spend minutes compiling 79 crates, or
# silently RELINK the artifact it was asked to merely identify, is not a
# resolver. Reading what the last build recorded is the strongest evidence
# available without becoming a build.
#
# THREE ORIGINS, and what each is allowed to do:
#   own       - dep-info (or .crates2.json) names THIS workspace. Usable.
#   foreign   - it names a DIFFERENT one. NEVER returned, in any pass.
#   unknown   - no such record. Usable, but only on the SHA evidence, exactly
#               as this file behaved before tree attribution existed. A host
#               whose cargo does not write dep-info therefore degrades to the
#               old behaviour rather than refusing everything.
# ---------------------------------------------------------------------------

apr_bin_die() {
    printf '%s\n' "$*" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Workspace facts, loaded at most once per shell.
#
# `cargo metadata` reports the real target directory under every checkout
# shape, which matters because `.cargo/config.toml` redirects it and is
# GITIGNORED (.gitignore:55) while its siblings `.cargo/audit.toml` and
# `.cargo/mutants.toml` are tracked. The directory looks version-controlled;
# the file that moves every build output is not. Measured:
#   main checkout  -> /mnt/nvme-raid0/coverage/aprender   (redirect applies)
#   fresh worktree -> <worktree>/target                   (no config, cargo default)
# A hardcoded absolute path is therefore right in exactly one of those and
# silently wrong in the other - which is how a release smoke-test came to read
# a five-hour-old binary and report a meaningless pass.
#
# The checkout is located via git, NOT via the script's own path. This used to
# derive the directory from `${BASH_SOURCE[0]}`, which is a BASH-ONLY variable.
# Sourcing this file from zsh - the interactive shell on the dev box - left it
# empty, so `dirname ""` gave `.`, the `cd ..` landed outside the checkout, and
# `cargo metadata` reported a DIFFERENT workspace's target dir. A resolver that
# silently resolves against the wrong workspace is the exact failure mode this
# file exists to prevent, so it must not depend on which shell sourced it.
# `git rev-parse --show-toplevel` is portable, and correct under worktrees.
# ---------------------------------------------------------------------------
apr_bin_load_meta() {
    if [ "${APR_BIN_META_LOADED:-0}" = "1" ]; then
        return 0
    fi
    APR_BIN_META_LOADED=1
    APR_BIN_TARGET_DIR=""
    APR_BIN_WS_ROOT=""
    APR_BIN_ANCHORS=""

    local here meta
    here=$(git rev-parse --show-toplevel 2>/dev/null) || here=""
    if [ -z "$here" ]; then
        here=$(pwd)
    fi
    # bashrs:allow SEC010 - $here is `git rev-parse --show-toplevel` or `pwd`,
    # never user-controlled input.
    # Every assignment from a command substitution in this file carries an
    # explicit `|| x=""`. A bare one is an errexit trigger, and this file is
    # SOURCED by callers running with `set -e`: a missing jq, or a `cargo` that
    # exits non-zero, would kill the caller's shell mid-source instead of
    # reaching the refusal at the bottom of this file.
    meta=$(cd "$here" && cargo metadata --no-deps --format-version 1 2>/dev/null) || meta=""
    if [ -z "$meta" ]; then
        return 1
    fi

    APR_BIN_TARGET_DIR=$(printf '%s\n' "$meta" | jq -r '.target_directory // empty' 2>/dev/null) || APR_BIN_TARGET_DIR=""
    APR_BIN_WS_ROOT=$(printf '%s\n' "$meta" | jq -r '.workspace_root // empty' 2>/dev/null) || APR_BIN_WS_ROOT=""
    # Every bin target named `apr` in this workspace, by absolute source path.
    # There is more than one: the root facade's src/bin/apr.rs and apr-cli's
    # own. Whichever produced the artifact, its path is in the dep-info.
    APR_BIN_ANCHORS=$(printf '%s\n' "$meta" \
        | jq -r '.packages[].targets[] | select(.name == "apr") | select(.kind | index("bin")) | .src_path' 2>/dev/null) || APR_BIN_ANCHORS=""
    return 0
}

# Kept as the published name other scripts may call.
apr_bin_target_dir() {
    apr_bin_load_meta || return 1
    printf '%s\n' "$APR_BIN_TARGET_DIR"
}

# The SECOND search root. `cargo metadata` answers for the shell asking, and
# the dev shell's `cargo` wrapper moves the target dir; a build launched from a
# plain bash script does not get that wrapper and lands in <worktree>/target.
# Searching only one of the two is how a tree's own binary becomes invisible
# while a sibling's is found in its place. Both are searched; attribution, not
# the search root, decides which one may be returned.
apr_bin_local_root() {
    apr_bin_load_meta || return 1
    if [ -z "$APR_BIN_WS_ROOT" ]; then
        return 1
    fi
    if [ "$APR_BIN_WS_ROOT/target" = "$APR_BIN_TARGET_DIR" ]; then
        return 1
    fi
    printf '%s\n' "$APR_BIN_WS_ROOT/target"
}

# Which tree built the binary that <bin>.d describes? Cosmetic only: it names
# the offending worktree in the report. The DECISION never depends on it.
apr_bin_dep_owner() {
    local dep="$1"
    if [ ! -f "$dep" ]; then
        return 1
    fi
    awk 'NR == 1 {
        for (i = 2; i <= NF; i++) {
            n = length($i)
            if (n > 15 && substr($i, n - 14) == "/src/bin/apr.rs") { print substr($i, 1, n - 15); exit }
            if (n > 26 && substr($i, n - 25) == "/crates/apr-cli/src/main.rs") { print substr($i, 1, n - 26); exit }
        }
    }' "$dep" 2>/dev/null
}

# Where was the bin named `apr` in $CARGO_HOME/bin installed FROM?
# Prints a directory for a `cargo install --path` install, nothing for a
# registry or git install (which cannot be this tree, and which the SHA check
# rejects on its own since a registry build embeds `+no-git`).
#
# Read out of .crates2.json rather than the older .crates.toml because jq is
# already a hard dependency of this file, and cargo writes that bin list inline
# for one binary and across four lines for two - a hand-rolled TOML parser for
# a load-bearing decision is exactly what this repo bans. `first(...)` rather
# than a pipe to `head`: an early-exiting reader makes the producer take
# SIGPIPE, and under `pipefail` the substitution then reports 141 though it
# matched.
apr_bin_installed_from() {
    local crates2="$1" key path
    if [ ! -f "$crates2" ]; then
        return 1
    fi
    key=$(jq -r 'first(.installs | to_entries[] | select(.value.bins | index("apr")) | .key) // empty' "$crates2" 2>/dev/null) || key=""
    case "$key" in
        *"path+file://"*) ;;
        *) return 1 ;;
    esac
    path="${key#*path+file://}"
    path="${path%%)*}"
    if [ -z "$path" ]; then
        return 1
    fi
    printf '%s\n' "$path"
}

# own | foreign | unknown. Never fails: an absent record is `unknown`, which is
# the pre-#2739 behaviour, not a refusal.
apr_bin_origin() {
    local bin="$1" dep inst
    apr_bin_load_meta || true

    dep="${bin}.d"
    if [ -f "$dep" ]; then
        if [ -z "$APR_BIN_ANCHORS" ]; then
            printf 'unknown\n'
            return 0
        fi
        # $APR_BIN_ANCHORS is newline-separated and POSIX grep treats a
        # multi-line pattern operand as one pattern per line, so this is
        # "matches ANY anchor" in a single call. Deliberately NOT a shell loop:
        # unquoted word splitting is what zsh does not do, and this file is
        # sourced from zsh. The emptiness guard above matters - `grep -F ""`
        # matches every line, which would attribute every binary to this tree.
        # The file operand keeps grep off a pipe, so `grep -q`'s early exit
        # cannot raise SIGPIPE in a caller running with `pipefail`.
        if grep -qF -- "$APR_BIN_ANCHORS" "$dep" 2>/dev/null; then
            printf 'own\n'
        else
            printf 'foreign\n'
        fi
        return 0
    fi

    case "$bin" in
        */bin/apr)
            inst=$(apr_bin_installed_from "${CARGO_HOME:-$HOME/.cargo}/.crates2.json") || inst=""
            if [ -n "$inst" ] && [ -n "$APR_BIN_WS_ROOT" ]; then
                case "$inst" in
                    "$APR_BIN_WS_ROOT"|"$APR_BIN_WS_ROOT"/*) printf 'own\n' ;;
                    *) printf 'foreign\n' ;;
                esac
                return 0
            fi
            ;;
    esac

    printf 'unknown\n'
    return 0
}

apr_bin_is_fresh() {
    local bin="$1" head
    head=$(git rev-parse --short HEAD 2>/dev/null) || head=""
    if [ -z "$head" ]; then
        return 1
    fi
    case "$("$bin" --version 2>&1)" in
        *"$head"*) return 0 ;;
    esac
    return 1
}

# One candidate, one mode. Prints the path and returns 0 on a match.
apr_bin_try() {
    local want="$1" cand="$2" origin
    if [ ! -x "$cand" ]; then
        return 1
    fi
    origin=$(apr_bin_origin "$cand")
    case "$want" in
        foreign)
            if [ "$origin" != "foreign" ]; then return 1; fi
            ;;
        own-fresh|own)
            if [ "$origin" != "own" ]; then return 1; fi
            ;;
        *)
            # THE FIX. A foreign binary is never returned, in any pass, at any
            # freshness. Two worktrees at the same commit built with different
            # features both satisfy the SHA check, so the SHA cannot be what
            # decides this.
            if [ "$origin" = "foreign" ]; then return 1; fi
            ;;
    esac
    case "$want" in
        own-fresh|any-fresh)
            if ! apr_bin_is_fresh "$cand"; then return 1; fi
            ;;
    esac
    printf '%s\n' "$cand"
    return 0
}

# The candidate SET, walked in a fixed order, fully quoted - no arrays (mini
# runs bash 3.2) and no word splitting (zsh does not split unquoted
# expansions). This deliberately does NOT consult PATH or any absolute path.
# Four `apr` binaries were found coexisting on one machine and every resolution
# strategy that *searches* for a binary eventually finds the wrong one; what is
# searched here is the small set of places cargo itself writes.
apr_bin_scan() {
    local want="$1" td lr ch
    apr_bin_load_meta || true
    td="$APR_BIN_TARGET_DIR"
    lr=$(apr_bin_local_root) || lr=""
    ch="${CARGO_HOME:-$HOME/.cargo}"

    if [ -n "$td" ]; then
        apr_bin_try "$want" "$td/release/apr" && return 0
        apr_bin_try "$want" "$td/debug/apr" && return 0
    fi
    if [ -n "$lr" ]; then
        apr_bin_try "$want" "$lr/release/apr" && return 0
        apr_bin_try "$want" "$lr/debug/apr" && return 0
    fi
    # Last resort: the `cargo install` destination. `cargo install` builds in a
    # temp dir and copies only the finished binary here, so it leaves nothing
    # in the target dirs above - qwen-story-daily installs exactly this way.
    # Kept LAST so a checkout that has built its own binary always tests that
    # one, and it is freshness- and origin-checked like every other candidate,
    # so this is a fallback in resolution order only, never a way around the
    # gate.
    apr_bin_try "$want" "$ch/bin/apr" && return 0
    return 1
}

apr_bin_report_candidate() {
    local cand="$1" origin owner
    if [ ! -x "$cand" ]; then
        return 0
    fi
    origin=$(apr_bin_origin "$cand")
    owner=$(apr_bin_dep_owner "${cand}.d") || owner=""
    if [ -z "$owner" ] && [ "$origin" != "own" ]; then
        case "$cand" in
            */bin/apr) owner=$(apr_bin_installed_from "${CARGO_HOME:-$HOME/.cargo}/.crates2.json") || owner="" ;;
        esac
    fi
    printf '    %-46s %-8s %s\n' "$cand" "$origin" "$("$cand" --version 2>&1 | head -1)"
    if [ -n "$owner" ] && [ "$origin" = "foreign" ]; then
        printf '      built from: %s\n' "$owner"
    fi
}

apr_bin_report_all_candidates() {
    local td lr ch
    apr_bin_load_meta || true
    td="$APR_BIN_TARGET_DIR"
    lr=$(apr_bin_local_root) || lr=""
    ch="${CARGO_HOME:-$HOME/.cargo}"
    if [ -n "$td" ]; then
        apr_bin_report_candidate "$td/release/apr"
        apr_bin_report_candidate "$td/debug/apr"
    fi
    if [ -n "$lr" ]; then
        apr_bin_report_candidate "$lr/release/apr"
        apr_bin_report_candidate "$lr/debug/apr"
    fi
    apr_bin_report_candidate "$ch/bin/apr"
}

# Name every `apr` on PATH so the shadowing is obvious rather than something
# the reader has to go discover. `command -v -a` is not valid bash (-a is not a
# `command` option); use `type -aP`, which lists every PATH match in resolution
# order.
apr_bin_report_path_shadows() {
    local shadows p
    shadows=$(type -aP apr 2>/dev/null | awk '!seen[$0]++') || shadows=""
    if [ -z "$shadows" ]; then
        return 0
    fi
    printf '  every apr on PATH - first wins:\n'
    printf '%s\n' "$shadows" | while IFS= read -r p; do
        if [ -n "$p" ]; then
            printf '    %-46s %s\n' "$p" "$("$p" --version 2>&1 | head -1)"
        fi
    done
}

# The diagnosis this file did not have. "Rebuild" is the fix for STALE and is
# useless here, so the two must never share a message.
apr_bin_report_wrong_tree() {
    local head
    head=$(git rev-parse --short HEAD 2>/dev/null) || head="<unknown>"
    {
        printf 'WRONG-TREE apr BINARY\n'
        printf '  this checkout : %s\n' "$APR_BIN_WS_ROOT"
        printf '  HEAD          : %s\n' "$head"
        printf '  Every apr binary this resolver can see was built from a\n'
        printf '  DIFFERENT source tree. It is not stale - there is no build of\n'
        printf '  this tree to be stale. Rebuilding somewhere else, or running\n'
        printf '  cargo install again, will not change that.\n'
        printf '  candidates:\n'
        apr_bin_report_all_candidates
        apr_bin_report_path_shadows
        printf '  why: concurrent worktrees of this repo share one target\n'
        printf '       directory, so whichever tree built last owns\n'
        printf '       <target>/release/apr. Check CARGO_TARGET_DIR, any\n'
        printf '       .cargo/config.toml build.target-dir, and any `cargo`\n'
        printf '       shell function before assuming your build landed here.\n'
        printf '  fix: build in THIS worktree, with the features you need\n'
        printf '         cargo build --release --bin apr\n'
        printf '       or give this worktree a target dir of its own\n'
        printf '         CARGO_TARGET_DIR=%s/target cargo build --release --bin apr\n' "$APR_BIN_WS_ROOT"
        printf '       or point the resolver at the binary you mean\n'
        printf '         APR_BIN=/path/to/apr\n'
    } >&2
}

apr_bin_resolve() {
    # Explicit override is the ONLY escape hatch - needed for A/B work such as
    # comparing a released binary against HEAD. It is still freshness-checked
    # below, so it cannot be used to smuggle a stale binary past the gate.
    if [ -n "${APR_BIN:-}" ]; then
        printf '%s\n' "$APR_BIN"
        return 0
    fi

    apr_bin_load_meta || true
    if [ -z "$APR_BIN_TARGET_DIR" ]; then
        return 1
    fi

    # Order is evidence-driven, not a fixed profile order. This file used to
    # return `release/apr` whenever it existed and hand it to the freshness
    # check, which refused it. Measured: with `debug/apr` built from HEAD
    # sitting beside a stale `release/apr`, it picked release and HARD-FAILED -
    # telling the caller to `cargo install` while a provably correct binary was
    # in the next directory. Every gate that sources this file broke that way,
    # and the trigger is just "you ran `cargo build --release` here once".
    #
    #   1. this tree, this commit          - the only fully proven case
    #   2. unattributable, this commit     - SHA evidence only, as before
    #   3. this tree, older commit         - honestly STALE, and says so
    #   4. unattributable, older commit    - STALE, as before
    # A foreign binary appears in none of these passes.
    apr_bin_scan own-fresh && return 0
    apr_bin_scan any-fresh && return 0
    apr_bin_scan own && return 0
    apr_bin_scan any && return 0

    # Nothing usable. If the only things on offer belong to another tree, say
    # THAT, rather than sending the reader to rebuild something that is not
    # stale. Reported here because this function already knows the candidates;
    # it writes to stderr, which command substitution does not capture.
    if apr_bin_scan foreign >/dev/null; then
        apr_bin_report_wrong_tree
        return 2
    fi
    return 1
}

# Assert the binary's embedded SHA matches HEAD. No-op outside a git checkout
# (an end user running a released apr has no HEAD to compare against).
apr_bin_assert_fresh() {
    local bin="$1"
    local reported head

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

    {
        printf 'STALE apr BINARY\n'
        printf '  resolved : %s\n' "$bin"
        printf '  reports  : %s\n' "$reported"
        printf '  HEAD     : %s\n' "$head"
        printf '  The binary was NOT built from this commit, so anything it\n'
        printf '  validates says nothing about this checkout.\n'
        printf '  candidates:\n'
        apr_bin_report_all_candidates
        apr_bin_report_path_shadows
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
#
# `cmd || rc=$?` rather than `rc=$?` after an `if`: inside `if ! cmd; then`,
# `$?` is the status of the negation, not of cmd, and a bare assignment from a
# command substitution is an errexit trigger in a caller running with `set -e`.
# This form is both errexit-exempt and reads the status that was meant.
# Re-measure on every source. The memo above is a per-source cache, not a
# per-shell one: a caller that sources this file, cd's into a different
# checkout and sources it again must get that checkout's answer, not the
# first one's.
APR_BIN_META_LOADED=0

APR_BIN_RC=0
APR=$(apr_bin_resolve) || APR_BIN_RC=$?

if [ "$APR_BIN_RC" -eq 2 ]; then
    # apr_bin_resolve already printed the WRONG-TREE report.
    return 1 2>/dev/null || exit 1
fi

if [ "$APR_BIN_RC" -ne 0 ]; then
    apr_bin_die "apr_bin: no apr binary found (build one: cargo build --release --bin apr, or cargo install --path crates/apr-cli --force)"
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
