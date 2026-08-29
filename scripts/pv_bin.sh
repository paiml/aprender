# pv_bin.sh — resolve THE pv built from THIS TREE at HEAD, and prove it.
#
# Source it, never execute it:
#     . scripts/pv_bin.sh || exit 1
#     "$PV" lint contracts/
#
# WHY THIS EXISTS. `pv` on PATH was 0.49.0 while the in-tree crate was 0.63.0,
# and the two disagreed on the gate that matters: strict-test-binding reported
# 253 refs / 51 missing under the stale binary and 371 / 27 under HEAD. Both
# surfaces where the RELEASE decision is made were using the stale one:
#   scripts/dogfood_surfaces.sh  printed `pv present (pv 0.49.0)` into the
#                                release receipt AS EVIDENCE OF CORRECTNESS
#   Makefile `contracts:`        ran a bare `pv lint contracts/` as the gate
#
# This is the same defect the repo already solved for `apr` (CLAUDE.md "Step 0 —
# pin the binary, ALWAYS"; four apr binaries once coexisted and a bare `apr`
# resolved to a 26-day-old copy). Same remedy, same shape as scripts/apr_bin.sh.
#
# CARGO IS THE FRESHNESS AUTHORITY, not a version string. A version match is not
# freshness: during this work pv was rebuilt three times at the SAME version with
# two distinct md5s, and the two binaries gave different answers on an identical
# tree. So we `cargo build` first and take the artifact cargo produces; the
# version assert below is a second line of defence against a PATH fallback, not
# the primary proof.
#
# OPTION-NEUTRAL BY CONSTRUCTION: this file sets no shell options. `set -euo
# pipefail` in a SOURCED file mutates the CALLER's shell — that leak once killed
# the nightly six lines in (see CLAUDE.md, scripts/check_sourced_libs_option_neutral.sh).
# Failure is signalled by RETURN STATUS only.
#
# ---------------------------------------------------------------------------
# A BUILD IS NOT AN ATTRIBUTION: TREE OWNERSHIP (#2745)
# ---------------------------------------------------------------------------
# Everything above proves WHICH VERSION a binary reports and that a build was
# attempted. It said nothing about WHICH TREE produced the artifact that was
# then picked up — and this file picked one by looking in a directory and
# taking the first executable it found there.
#
# That directory is shared. The dev shell defines a `cargo` FUNCTION exporting
#     CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/$(basename <remote.origin.url>)
# keyed on the remote URL, so all 40+ git worktrees of this repo build into ONE
# target directory and whichever tree built last owns <target>/debug/pv. The
# function exists only in the interactive shell, so a build launched from a
# `#!/usr/bin/env bash` script gets the real cargo and lands in
# <workspace_root>/target instead. Measured:
#
#   interactive (function):  /mnt/nvme-raid0/targets/aprender
#   via bash script:         /home/noah/src/aprender/target
#
# The same tree builds to two places depending on who launched the build, and
# `cargo metadata` answers for the shell ASKING rather than the shell that
# BUILT. Measured live from worktree t3-pvbin, with the resolver's two search
# candidates read straight out of cargo's own dep-info:
#
#   /mnt/nvme-raid0/targets/aprender/debug/pv.d
#       .../.claude/worktrees/wf_9b8aff2c-325-4/crates/aprender-contracts-cli/src/main.rs
#   /mnt/nvme-raid0/targets/aprender/release/pv.d
#       /home/noah/src/aprender/crates/aprender-contracts-cli/src/main.rs
#   ~/.cargo/.crates2.json
#       aprender-contracts-cli 0.63.0
#       (path+file:///home/noah/src/aprender/crates/aprender-contracts-cli)
#
# Neither artifact belongs to the tree that was asking, and `release/pv` — the
# candidate this file fell through to whenever `debug/pv` was absent — is never
# written by the build above at all. It is simply whatever some other worktree
# left there.
#
# WHY THE VERSION CHECK CANNOT COVER THIS. Unlike `apr --version`, `pv
# --version` carries NO git sha: it prints a semver plus an identity marker,
# and the semver is a WORKSPACE version shared by all 40+ worktrees. Two trees
# at the same release therefore print byte-identical version lines. The
# strongest check this file had was satisfied, in full, by another tree's
# binary — and it would have said nothing. A pv resolved that way validates
# contracts with a stranger's verifier and reports a clean gate.
#
# HOW A BINARY IS ATTRIBUTED TO A TREE. Cargo writes a dep-info file next to
# every binary it links: <target>/<profile>/pv.d, a make-style rule whose
# right-hand side lists the ABSOLUTE PATH of every source that went into it.
# That is cargo's own record of which tree it compiled, written in the same
# step that writes the binary, plain text on every host. The anchor compared
# against it is the `src_path` cargo metadata reports for the bin target named
# `pv` in THIS workspace, so both sides of the comparison are cargo's own path
# normalisation and cannot disagree about symlinks.
#
# The anchor is matched WHOLE, never as a workspace-root prefix. This repo's
# agent worktrees live UNDER the main checkout
# (/home/noah/src/aprender/.claude/worktrees/wf_...), so a prefix test would
# attribute every one of them to the main checkout; and a worktree's dep-info
# legitimately contains the MAIN checkout's `.git/worktrees/...` paths, which a
# prefix test reads as proof of ownership. The full src_path is unambiguous.
#
# `cargo install` leaves no dep-info — it builds in a temp dir and copies only
# the finished binary — so that candidate is attributed from
# $CARGO_HOME/.crates2.json, which records the source the bins were installed
# FROM. That comparison is EXACT against the manifest directory of the package
# owning the `pv` bin target, for the same reason: `cargo install --path
# <main>/.claude/worktrees/X/crates/aprender-contracts-cli` is a different tree
# from `<main>`, and a prefix test would call it ours.
#
# A registry or git install (`registry+...`, `git+...`) is FOREIGN outright.
# apr_bin.sh can afford to leave that case unattributed because a registry
# build embeds `+no-git` and its sha check rejects it; pv has no sha, and a
# crates.io `aprender-contracts-cli` at the declared version prints exactly
# what this tree's build prints. `cargo install aprender-contracts-cli` is a
# documented, advertised route (crates/facades/provable-contracts-cli), so this
# is the ordinary case, not a corner.
#
# WHY NOT JUST ASK CARGO WHICH FILE IT WROTE. `cargo build --bin pv
# --message-format=json` prints the executable path authoritatively, and is the
# obviously right answer — for a build script. It is wrong here: this file is
# SOURCED by every contract gate in the repo, and a resolver that may relink
# the artifact it was asked to merely identify, or spend minutes in a
# dependency graph, is not a resolver. The `cargo build -q` above stays because
# it is this file's declared freshness authority and every caller depends on a
# pv existing afterwards; reading what that build recorded is the strongest
# evidence available without ALSO becoming the thing that decides.
#
# THREE ORIGINS, and what each is allowed to do:
#   own       — dep-info (or .crates2.json) names THIS workspace. Usable.
#   foreign   — it names a DIFFERENT one. NEVER returned, in any pass.
#   unknown   — no such record. Usable, but only on the version+identity
#               evidence, exactly as this file behaved before tree attribution
#               existed. A host whose cargo does not write dep-info therefore
#               degrades to the old behaviour rather than refusing everything.
#
# STALE and WRONG-TREE are reported separately and never together. "Rebuild" is
# the remedy for one and useless for the other.
# ---------------------------------------------------------------------------

pv_bin_die() {
    printf 'pv_bin: %s\n' "$*" >&2
    return 1
}

# The identity marker `pv --version` carries since #2559, held in ONE place
# because two decisions read it: the scan's freshness predicate and the final
# assert. Four things claim the name `pv` — pv(1) the pipe viewer, the crates.io
# `pv` crate, the deprecated facade, and this tool — and the operator settled
# 2026-08-21 that this binary keeps the name, which makes this string the whole
# mitigation.
PV_BIN_IDENTITY='(aprender provable-contracts verifier)'

pv_bin_root() {
    git rev-parse --show-toplevel 2>/dev/null || pwd
}

# ---------------------------------------------------------------------------
# Workspace facts, measured once per source.
#
# Every assignment from a command substitution carries an explicit `|| x=""`.
# A bare one is an errexit trigger, and this file is SOURCED by callers running
# with `set -e` (and by scripts/dogfood-book.sh under `set -u`): a missing jq,
# or a `cargo` that exits non-zero, would kill the caller's shell mid-source
# instead of reaching the refusal at the bottom of this file.
# ---------------------------------------------------------------------------
pv_bin_load_meta() {
    if [ "${PV_BIN_META_LOADED:-0}" = "1" ]; then
        return 0
    fi
    PV_BIN_META_LOADED=1
    PV_BIN_TARGET_DIR=""
    PV_BIN_WS_ROOT=""
    PV_BIN_ANCHORS=""
    PV_BIN_INSTALL_DIRS=""

    pv_bin_meta_here=$(pv_bin_root) || pv_bin_meta_here=""
    if [ -z "$pv_bin_meta_here" ]; then
        pv_bin_meta_here=$PWD
    fi
    pv_bin_meta_json=$( cd "$pv_bin_meta_here" && cargo metadata --no-deps --format-version 1 2>/dev/null ) || pv_bin_meta_json=""
    if [ -z "$pv_bin_meta_json" ]; then
        return 1
    fi

    PV_BIN_TARGET_DIR=$(printf '%s\n' "$pv_bin_meta_json" | jq -r '.target_directory // empty' 2>/dev/null) || PV_BIN_TARGET_DIR=""
    PV_BIN_WS_ROOT=$(printf '%s\n' "$pv_bin_meta_json" | jq -r '.workspace_root // empty' 2>/dev/null) || PV_BIN_WS_ROOT=""
    # Every bin target named `pv` in this workspace, by absolute source path.
    # There is exactly one today (crates/aprender-contracts-cli/src/main.rs);
    # the facade crate that also declared `[[bin]] pv` is workspace-EXCLUDED
    # and ships no binary since #2553. Written as a set anyway so a second one
    # does not silently become unattributable.
    PV_BIN_ANCHORS=$(printf '%s\n' "$pv_bin_meta_json" \
        | jq -r '.packages[].targets[] | select(.name == "pv") | select(.kind | index("bin")) | .src_path' 2>/dev/null) || PV_BIN_ANCHORS=""
    # The directory `cargo install --path DIR` would be given to install `pv`
    # from this tree — the manifest dir of the package owning that bin target.
    # This is what .crates2.json records, and the ONLY thing that attributes an
    # installed binary to a tree.
    PV_BIN_INSTALL_DIRS=$(printf '%s\n' "$pv_bin_meta_json" \
        | jq -r '.packages[] | select(any(.targets[]; .name == "pv" and (.kind | index("bin")))) | .manifest_path | sub("/Cargo\\.toml$"; "")' 2>/dev/null) || PV_BIN_INSTALL_DIRS=""
    return 0
}

# The version the TREE declares, memoised. `version.workspace = true` in the
# crate manifest does not match the `^version =` form on purpose, so the read
# falls through to the workspace manifest, which is where the number lives.
pv_bin_declared_version() {
    if [ -n "${PV_BIN_DECLARED:-}" ]; then
        printf '%s\n' "$PV_BIN_DECLARED"
        return 0
    fi
    pv_bin_dv_root=$(pv_bin_root) || pv_bin_dv_root="$PWD"
    PV_BIN_DECLARED=$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' \
        "$pv_bin_dv_root/crates/aprender-contracts-cli/Cargo.toml" 2>/dev/null) || PV_BIN_DECLARED=""
    if [ -z "$PV_BIN_DECLARED" ]; then
        PV_BIN_DECLARED=$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' \
            "$pv_bin_dv_root/Cargo.toml" 2>/dev/null) || PV_BIN_DECLARED=""
    fi
    if [ -z "$PV_BIN_DECLARED" ]; then
        return 1
    fi
    printf '%s\n' "$PV_BIN_DECLARED"
}

# The SECOND search root. `cargo metadata` answers for the shell asking, and the
# dev shell's `cargo` wrapper moves the target dir; a build launched from a
# plain bash script does not get that wrapper and lands in <worktree>/target.
# Searching only one of the two is how a tree's own binary becomes invisible
# while a sibling's is found in its place. Both are searched; ATTRIBUTION, not
# the search root, decides which one may be returned.
pv_bin_local_root() {
    pv_bin_load_meta || return 1
    if [ -z "${PV_BIN_WS_ROOT:-}" ]; then
        return 1
    fi
    if [ "$PV_BIN_WS_ROOT/target" = "${PV_BIN_TARGET_DIR:-}" ]; then
        return 1
    fi
    printf '%s\n' "$PV_BIN_WS_ROOT/target"
}

# Which tree built the binary that <bin>.d describes? Cosmetic only: it names
# the offending worktree in the report. The DECISION never depends on it.
# "/crates/aprender-contracts-cli/src/main.rs" is 42 characters.
pv_bin_dep_owner() {
    pv_bin_do_dep="$1"
    if [ ! -f "$pv_bin_do_dep" ]; then
        return 1
    fi
    awk 'NR == 1 {
        for (i = 2; i <= NF; i++) {
            n = length($i)
            if (n > 42 && substr($i, n - 41) == "/crates/aprender-contracts-cli/src/main.rs") {
                print substr($i, 1, n - 42); exit
            }
        }
    }' "$pv_bin_do_dep" 2>/dev/null
}

# The .crates2.json KEY of the install that owns a bin named `pv`, verbatim:
#   "aprender-contracts-cli 0.63.0 (path+file:///home/noah/src/aprender/crates/aprender-contracts-cli)"
# Read out of .crates2.json rather than the older .crates.toml because jq is
# already a hard dependency of this file, and cargo writes that bin list inline
# for one binary and across several lines for more — a hand-rolled TOML parser
# for a load-bearing decision is exactly what this repo bans.
#
# `first(...)` rather than a pipe to `head`: an early-exiting reader makes the
# producer take SIGPIPE, and under `pipefail` the substitution then reports 141
# though it matched.
pv_bin_installed_key() {
    pv_bin_ik_file="$1"
    if [ ! -f "$pv_bin_ik_file" ]; then
        return 1
    fi
    pv_bin_ik_key=$(jq -r 'first(.installs | to_entries[] | select(.value.bins | index("pv")) | .key) // empty' "$pv_bin_ik_file" 2>/dev/null) || pv_bin_ik_key=""
    if [ -z "$pv_bin_ik_key" ]; then
        return 1
    fi
    printf '%s\n' "$pv_bin_ik_key"
}

# The directory a `cargo install --path` key was installed FROM. Returns
# non-zero for a registry or git key, which carries no local directory at all.
pv_bin_install_source_dir() {
    case "$1" in
        *"path+file://"*) ;;
        *) return 1 ;;
    esac
    pv_bin_isd="${1#*path+file://}"
    pv_bin_isd="${pv_bin_isd%%)*}"
    pv_bin_isd="${pv_bin_isd%%\#*}"
    if [ -z "$pv_bin_isd" ]; then
        return 1
    fi
    printf '%s\n' "$pv_bin_isd"
}

# Is $1 one of the newline-separated paths in $2? WHOLE-LINE equality, never a
# prefix: this repo's worktrees live under the main checkout
# (/home/noah/src/aprender/.claude/worktrees/wf_...), so a prefix test would
# attribute every one of them to the main checkout.
#
# The list is fed to grep as a HERE-STRING, never as `printf ... | grep -qxF`.
# In the pipe form grep exits on its first match, the producer takes SIGPIPE,
# and under `pipefail` -- which the Makefile turns on for every recipe
# (.SHELLFLAGS := -o pipefail -c, and `make contracts` sources this file) --
# the pipeline reports 141 although grep MATCHED. A here-string has no producer
# process, so there is nothing to signal.
pv_bin_path_listed() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        return 1
    fi
    grep -qxF -- "$1" <<< "$2"
}

# own | foreign | unknown. Never fails: an absent record is `unknown`, which is
# the pre-#2745 behaviour, not a refusal.
pv_bin_origin() {
    pv_bin_or_bin="$1"
    pv_bin_load_meta || true

    pv_bin_or_dep="${pv_bin_or_bin}.d"
    if [ -f "$pv_bin_or_dep" ]; then
        if [ -z "${PV_BIN_ANCHORS:-}" ]; then
            printf 'unknown\n'
            return 0
        fi
        # $PV_BIN_ANCHORS is newline-separated and POSIX grep treats a
        # multi-line pattern operand as one pattern per line, so this is
        # "matches ANY anchor" in a single call. Deliberately NOT a shell loop:
        # unquoted word splitting is what zsh does not do, and this file is
        # sourced from zsh. The emptiness guard above matters — `grep -F ""`
        # matches every line, which would attribute every binary to this tree.
        # The file operand keeps grep off a pipe, so `grep -q`'s early exit
        # cannot raise SIGPIPE in a caller running with `pipefail`.
        if grep -qF -- "$PV_BIN_ANCHORS" "$pv_bin_or_dep" 2>/dev/null; then
            printf 'own\n'
        else
            printf 'foreign\n'
        fi
        return 0
    fi

    case "$pv_bin_or_bin" in
        */bin/pv)
            pv_bin_or_key=$(pv_bin_installed_key "${CARGO_HOME:-${HOME:-}/.cargo}/.crates2.json") || pv_bin_or_key=""
            if [ -n "$pv_bin_or_key" ]; then
                pv_bin_or_dir=$(pv_bin_install_source_dir "$pv_bin_or_key") || pv_bin_or_dir=""
                if [ -z "$pv_bin_or_dir" ]; then
                    # registry+ / git+ : cargo built this from a published
                    # crate or a remote ref. It is definitively not this tree,
                    # and pv carries no sha that would notice.
                    printf 'foreign\n'
                    return 0
                fi
                if pv_bin_path_listed "$pv_bin_or_dir" "${PV_BIN_INSTALL_DIRS:-}"; then
                    printf 'own\n'
                else
                    printf 'foreign\n'
                fi
                return 0
            fi
            ;;
    esac

    printf 'unknown\n'
    return 0
}

# Quiet freshness predicate: the version the binary reports must be the version
# the tree declares, AND the first line must carry the identity marker. Same
# two facts pv_bin_assert_fresh reports on; asserted here so the SCAN can pass
# over an old build instead of returning it and hard-failing.
pv_bin_version_ok() {
    pv_bin_vok_bin="$1"
    pv_bin_vok_declared=$(pv_bin_declared_version) || pv_bin_vok_declared=""
    if [ -z "$pv_bin_vok_declared" ]; then
        return 1
    fi
    pv_bin_vok_all=$("$pv_bin_vok_bin" --version 2>&1) || pv_bin_vok_all=""
    pv_bin_vok_first=$(printf '%s\n' "$pv_bin_vok_all" | awk 'NR==1{print}') || pv_bin_vok_first=""
    pv_bin_vok_semver=$(printf '%s\n' "$pv_bin_vok_all" | awk 'NR==1{print $2; exit}') || pv_bin_vok_semver=""
    if [ "$pv_bin_vok_semver" != "$pv_bin_vok_declared" ]; then
        return 1
    fi
    case "$pv_bin_vok_first" in
        *"$PV_BIN_IDENTITY"*) return 0 ;;
    esac
    return 1
}

# One candidate, one mode. Prints the path and returns 0 on a match.
pv_bin_try() {
    pv_bin_try_want="$1"
    pv_bin_try_cand="$2"
    # -f as well as -x: `[ -x DIR ]` is TRUE for any searchable directory, so a
    # stray directory named `pv` would be "found" and then fail to run.
    if [ ! -f "$pv_bin_try_cand" ] || [ ! -x "$pv_bin_try_cand" ]; then
        return 1
    fi
    pv_bin_try_origin=$(pv_bin_origin "$pv_bin_try_cand") || pv_bin_try_origin="unknown"
    case "$pv_bin_try_want" in
        foreign)
            if [ "$pv_bin_try_origin" != "foreign" ]; then return 1; fi
            ;;
        own-fresh|own)
            if [ "$pv_bin_try_origin" != "own" ]; then return 1; fi
            ;;
        *)
            # THE FIX. A foreign binary is never returned, in any pass, at any
            # freshness. Two worktrees at the same workspace version print
            # byte-identical `pv --version` output, so the version cannot be
            # what decides this.
            if [ "$pv_bin_try_origin" = "foreign" ]; then return 1; fi
            ;;
    esac
    case "$pv_bin_try_want" in
        own-fresh|any-fresh)
            if ! pv_bin_version_ok "$pv_bin_try_cand"; then return 1; fi
            ;;
    esac
    printf '%s\n' "$pv_bin_try_cand"
    return 0
}

# The candidate SET, walked in a fixed order, fully quoted — no arrays (mini
# runs bash 3.2) and no word splitting (zsh does not split unquoted
# expansions). `debug` precedes `release` in each root because the build above
# is a debug build; `release/pv` is never written by this file and is therefore
# whatever some other build left behind, which is exactly why it is attributed
# rather than trusted.
pv_bin_scan() {
    pv_bin_scan_want="$1"
    pv_bin_load_meta || true
    pv_bin_scan_td="${PV_BIN_TARGET_DIR:-}"
    pv_bin_scan_lr=$(pv_bin_local_root) || pv_bin_scan_lr=""
    pv_bin_scan_ch="${CARGO_HOME:-${HOME:-}/.cargo}"

    if [ -n "$pv_bin_scan_td" ]; then
        pv_bin_try "$pv_bin_scan_want" "$pv_bin_scan_td/debug/pv" && return 0
        pv_bin_try "$pv_bin_scan_want" "$pv_bin_scan_td/release/pv" && return 0
    fi
    if [ -n "$pv_bin_scan_lr" ]; then
        pv_bin_try "$pv_bin_scan_want" "$pv_bin_scan_lr/debug/pv" && return 0
        pv_bin_try "$pv_bin_scan_want" "$pv_bin_scan_lr/release/pv" && return 0
    fi
    # Last resort: the `cargo install` destination. `cargo install` builds in a
    # temp dir and copies only the finished binary here, so it leaves nothing
    # in the target dirs above. Kept LAST so a checkout that built its own pv
    # always tests that one, and it is origin- and version-checked like every
    # other candidate — a fallback in resolution ORDER only, never a way around
    # the gate.
    pv_bin_try "$pv_bin_scan_want" "$pv_bin_scan_ch/bin/pv" && return 0
    return 1
}

pv_bin_report_candidate() {
    pv_bin_rc_cand="$1"
    if [ ! -f "$pv_bin_rc_cand" ] || [ ! -x "$pv_bin_rc_cand" ]; then
        return 0
    fi
    pv_bin_rc_origin=$(pv_bin_origin "$pv_bin_rc_cand") || pv_bin_rc_origin="unknown"
    pv_bin_rc_owner=$(pv_bin_dep_owner "${pv_bin_rc_cand}.d") || pv_bin_rc_owner=""
    if [ -z "$pv_bin_rc_owner" ] && [ "$pv_bin_rc_origin" != "own" ]; then
        case "$pv_bin_rc_cand" in
            */bin/pv) pv_bin_rc_owner=$(pv_bin_installed_key "${CARGO_HOME:-${HOME:-}/.cargo}/.crates2.json") || pv_bin_rc_owner="" ;;
        esac
    fi
    pv_bin_rc_ver=$("$pv_bin_rc_cand" --version 2>&1) || pv_bin_rc_ver=""
    pv_bin_rc_first=$(printf '%s\n' "$pv_bin_rc_ver" | awk 'NR==1{print}') || pv_bin_rc_first=""
    printf '    %-46s %-8s %s\n' "$pv_bin_rc_cand" "$pv_bin_rc_origin" "$pv_bin_rc_first"
    if [ -n "$pv_bin_rc_owner" ] && [ "$pv_bin_rc_origin" = "foreign" ]; then
        printf '      built from: %s\n' "$pv_bin_rc_owner"
    fi
}

pv_bin_report_all_candidates() {
    pv_bin_load_meta || true
    pv_bin_rac_td="${PV_BIN_TARGET_DIR:-}"
    pv_bin_rac_lr=$(pv_bin_local_root) || pv_bin_rac_lr=""
    pv_bin_rac_ch="${CARGO_HOME:-${HOME:-}/.cargo}"
    if [ -n "$pv_bin_rac_td" ]; then
        pv_bin_report_candidate "$pv_bin_rac_td/debug/pv"
        pv_bin_report_candidate "$pv_bin_rac_td/release/pv"
    fi
    if [ -n "$pv_bin_rac_lr" ]; then
        pv_bin_report_candidate "$pv_bin_rac_lr/debug/pv"
        pv_bin_report_candidate "$pv_bin_rac_lr/release/pv"
    fi
    pv_bin_report_candidate "$pv_bin_rac_ch/bin/pv"
}

# Name every `pv` on PATH so the shadowing is obvious rather than something the
# reader has to go discover. This matters more for pv than for apr: pv(1) the
# pipe viewer is a normal package on these hosts and is usually FIRST.
#
# PATH is split with tr rather than `type -aP` (bash-only; this file is sourced
# from zsh, where `-P` is not a `type` option) and rather than word splitting
# (which zsh does not do on unquoted expansions). The awk dedup is not cosmetic
# on this box: ~/.cargo/bin appears twice in PATH, and a list that prints the
# same path twice reads like two different binaries.
pv_bin_report_path_shadows() {
    # Every stage reads its producer to EOF -- `tr` and the `while` loop both
    # consume everything, and the awk below has no `exit` -- so no reader can
    # close a pipe early and hand its producer a SIGPIPE that `pipefail` would
    # report as 141.
    pv_bin_ps_list=$(
        printf '%s\n' "${PATH:-}" | tr ':' '\n' | awk '!seen[$0]++' | while IFS= read -r pv_bin_ps_dir; do
            if [ -z "$pv_bin_ps_dir" ] || [ ! -f "$pv_bin_ps_dir/pv" ] || [ ! -x "$pv_bin_ps_dir/pv" ]; then
                continue
            fi
            pv_bin_ps_ver=$("$pv_bin_ps_dir/pv" --version 2>&1) || pv_bin_ps_ver=""
            pv_bin_ps_first=$(printf '%s\n' "$pv_bin_ps_ver" | awk 'NR==1{print}') || pv_bin_ps_first=""
            printf '    %-46s %s\n' "$pv_bin_ps_dir/pv" "$pv_bin_ps_first"
        done
    ) || pv_bin_ps_list=""
    if [ -z "$pv_bin_ps_list" ]; then
        return 0
    fi
    printf '  every pv on PATH - first wins:\n'
    printf '%s\n' "$pv_bin_ps_list"
    return 0
}

# The diagnosis this file did not have. "Rebuild" is the fix for STALE and is
# useless here, so the two must never share a message — and must never appear
# in the same report, because emitting both is emitting neither.
pv_bin_report_wrong_tree() {
    pv_bin_declared_version >/dev/null 2>&1 || true
    {
        printf 'WRONG-TREE pv BINARY\n'
        printf '  this checkout : %s\n' "${PV_BIN_WS_ROOT:-<unknown>}"
        printf '  declares      : %s\n' "${PV_BIN_DECLARED:-<unknown>}"
        printf '  Every pv binary this resolver can see was built from a\n'
        printf '  DIFFERENT source tree, so its verdict on these contracts is\n'
        printf '  a verdict about other code. It is not out of date - there is\n'
        printf '  no build of this tree for it to be behind. Running the build\n'
        printf '  again somewhere else, or cargo install, will not change that.\n'
        printf '  candidates:\n'
        pv_bin_report_all_candidates
        pv_bin_report_path_shadows
        printf '  why: concurrent worktrees of this repo share one target\n'
        printf '       directory, so whichever tree built last owns\n'
        printf '       <target>/debug/pv. Check CARGO_TARGET_DIR, any\n'
        printf '       .cargo/config.toml build.target-dir, and any `cargo`\n'
        printf '       shell function before assuming your build landed here.\n'
        printf '  fix: build in THIS worktree\n'
        printf '         cargo build -p aprender-contracts-cli --bin pv\n'
        printf '       or give this worktree a target dir of its own\n'
        printf '         CARGO_TARGET_DIR=%s/target cargo build -p aprender-contracts-cli --bin pv\n' "${PV_BIN_WS_ROOT:-.}"
        printf '       or point the resolver at the binary you mean\n'
        printf '         PV_BIN=/path/to/pv\n'
    } >&2
}

# Build from HEAD, then hand back an artifact that is provably THIS tree's.
pv_bin_resolve() {
    # Explicit override is the ONLY escape hatch — needed for A/B work such as
    # comparing a released pv against HEAD. It is still version- and
    # identity-checked below, so it cannot smuggle a stale binary past the gate.
    if [ -n "${PV_BIN:-}" ]; then
        printf '%s\n' "$PV_BIN"
        return 0
    fi
    pv_bin_res_root=$(pv_bin_root) || pv_bin_res_root="$PWD"
    ( cd "$pv_bin_res_root" && cargo build -q -p aprender-contracts-cli --bin pv ) >&2 \
        || { pv_bin_die "cargo build of aprender-contracts-cli failed"; return 1; }

    pv_bin_load_meta || true
    if [ -z "${PV_BIN_TARGET_DIR:-}" ]; then
        pv_bin_die "could not read cargo target_directory"
        return 1
    fi

    # Order is evidence-driven, not a fixed profile order.
    #   1. this tree, this version        — the only fully proven case
    #   2. unattributable, this version   — version+identity evidence only
    #   3. this tree, older version       — honestly STALE, and says so
    #   4. unattributable, older version  — STALE, as before
    # A foreign binary appears in none of these passes.
    pv_bin_scan own-fresh && return 0
    pv_bin_scan any-fresh && return 0
    pv_bin_scan own && return 0
    pv_bin_scan any && return 0

    # Nothing usable. If the only things on offer belong to another tree, say
    # THAT rather than sending the reader to rebuild something that is not out
    # of date. Reported here because this function already knows the
    # candidates; it writes to stderr, which command substitution does not
    # capture.
    if pv_bin_scan foreign >/dev/null; then
        pv_bin_report_wrong_tree
        return 2
    fi
    pv_bin_die "no pv binary under $PV_BIN_TARGET_DIR after a successful build"
    return 1
}

# Second line of defence: the resolved binary must report the version the tree
# declares, and must identify itself. Catches a PATH fallback or a hand-copied
# artifact, and is the check the PV_BIN override still has to satisfy.
pv_bin_assert_fresh() {
    pv_bin_b="$1"
    [ -f "$pv_bin_b" ] && [ -x "$pv_bin_b" ] || { pv_bin_die "not executable: $pv_bin_b"; return 1; }
    pv_bin_declared=$(pv_bin_declared_version) || pv_bin_declared=""
    [ -n "$pv_bin_declared" ] || { pv_bin_die "could not read declared pv version"; return 1; }
    # POSITIONAL, first line only. `pv --version` is deliberately multi-line as
    # of #2559 — it has to say WHICH pv this is, because four things claim that
    # name. The old `awk '{print $NF}'` took the last field of EVERY line and
    # handed this comparison four lines of prose. The version line's shape is
    # pinned from the other side by
    # crates/aprender-contracts-cli/tests/version_identity.rs
    # (`semver_stays_the_second_field_of_the_first_line`), and by the case table
    # in scripts/check_pv_version_parse.sh.
    # ONE invocation, then both reads off the SAME captured text -- the semver
    # and the identity must describe the same binary and the same run.
    pv_bin_vers=$("$pv_bin_b" --version 2>&1) || pv_bin_vers=""
    pv_bin_actual=$(printf '%s\n' "$pv_bin_vers" | awk 'NR==1{print $2; exit}') || pv_bin_actual=""
    pv_bin_first=$(printf '%s\n' "$pv_bin_vers" | awk 'NR==1{print}') || pv_bin_first=""
    if [ "$pv_bin_actual" != "$pv_bin_declared" ]; then
        {
            printf 'STALE pv BINARY\n'
            printf '  resolved : %s\n' "$pv_bin_b"
            printf '  reports  : %s\n' "$pv_bin_actual"
            printf '  declares : %s\n' "$pv_bin_declared"
            printf '  It reports a version this tree does not declare, so what\n'
            printf '  it says about these contracts is about other code.\n'
            printf '  candidates:\n'
            pv_bin_report_all_candidates
            pv_bin_report_path_shadows
            printf '  fix: cargo build -p aprender-contracts-cli --bin pv\n'
            printf '       or set PV_BIN to the binary you mean.\n'
        } >&2
        return 1
    fi
    # IDENTITY, on the same line. #2559 added an identity string to `pv
    # --version` precisely because four things claim the name `pv` -- but this
    # function, the place the release actually DECIDES whether it has the right
    # binary, went on proving freshness from the SEMVER alone. A semver is not
    # an identity: pv(1) the pipe viewer ships 0.x versions too, and a stale
    # sibling `pv` that happened to match the declared version would have
    # satisfied the check above unchanged. So the marker is asserted here, where
    # the decision is made, not only in the unit tests that watch the string.
    case "$pv_bin_first" in
        *"$PV_BIN_IDENTITY"*) ;;
        *)
            pv_bin_die "resolved pv does not identify as the aprender provable-contracts verifier. First --version line: $pv_bin_first ($pv_bin_b). If that is pv(1) the pipe viewer, or another crate named pv, it is the wrong binary however its version reads."
            return 1
            ;;
    esac
    return 0
}

# Re-measure on every source. The memos above are per-SOURCE caches, not
# per-shell ones: a caller that sources this file, cd's into a different
# checkout and sources it again must get that checkout's answer, not the first
# one's.
PV_BIN_META_LOADED=0
PV_BIN_DECLARED=""

PV_BIN_RC=0
PV=$(pv_bin_resolve) || PV_BIN_RC=$?

if [ "$PV_BIN_RC" -eq 2 ]; then
    # pv_bin_resolve already printed the WRONG-TREE report; adding a second
    # diagnosis here is how a report comes to say STALE and WRONG-TREE at once.
    return 1 2>/dev/null || exit 1
fi
if [ "$PV_BIN_RC" -ne 0 ]; then
    return 1 2>/dev/null || exit 1
fi
pv_bin_assert_fresh "$PV" || return 1 2>/dev/null || exit 1
export PV
