# shellcheck shell=bash
# lib_code_identity.sh — the ONE definition of "same code" (#4673, BLD-002 row R1).
#
# Contract: contracts/code-identity-v1.yaml. Case table and planted mutants:
# scripts/release/check_code_identity.sh.
#
# The operator ruled the identity is the TREE: every tracked file except the root
# evidence/ path. Recording a result never changes H; any other edit does.
#
#   H(REV) = sha256 over REV's tracked entries "<mode> <type> <object>\t<path>\0"
#            (git ls-tree -r -z --full-tree order), dropping the root path
#            `evidence` and every path under `evidence/`.
#   same(A, B)  <=>  H(A) = H(B)  <=>  git diff --quiet A B -- . ':(exclude)evidence'
#
# H reads only committed objects: never the worktree, the index or untracked files.
# The mode is hashed, so a chmod +x is a code change.
#
# This file is SOURCED. It sets no shell options and fails by return status:
#   0 = measured (same / identity printed), 1 = measured and different,
#   2 = not_measured (unknown rev, git failure, tool missing). 2 is never a pass.
#
#   . scripts/release/lib_code_identity.sh || exit 1
#   code_identity REV [ROOT]           -> prints "H=<64 hex>"; rc 0 or 2
#   code_identity_same A B [ROOT]      -> rc 0 same, 1 different, 2 not_measured
#   code_identity_bump_of M B NEWVER   -> rc 0 iff B is exactly M + bump-version.sh NEWVER
# ROOT is the repository to read (default: the current directory).

# _code_identity_hash TREEISH [GIT-ARGS...] -> prints the 64-hex H; rc 0 or 2
_code_identity_hash() {
    local treeish="$1" h
    shift
    # One entry per NUL-terminated record "<meta>\t<path>"; <meta> holds no tab. grep -z
    # drops the root `evidence` path and everything under `evidence/`; grep exits 1 when
    # every entry was dropped, which is still a measurement, and 2 on an error.
    h="$(
        git "$@" ls-tree -r -z --full-tree "$treeish" 2>/dev/null |
            LC_ALL=C grep -zvE $'^[^\t]*\tevidence(/|$)' |
            sha256sum
        st=("${PIPESTATUS[@]}")
        [ "${st[0]}" = 0 ] && [ "${st[1]}" -le 1 ] && [ "${st[2]}" = 0 ] || exit 2
    )" || return 2
    h="${h%% *}"
    case "$h" in
        '' | *[!0-9a-f]*) return 2 ;;
    esac
    [ "${#h}" = 64 ] || return 2
    printf '%s\n' "$h"
}

# code_identity REV [ROOT] -> prints "H=<64 hex>"; rc 0, or 2 when REV is not a commit in ROOT
code_identity() {
    local rev="${1:-}" root="${2:-.}" c h
    [ -n "$rev" ] || return 2
    c="$(git -C "$root" rev-parse --verify --quiet "${rev}^{commit}" 2>/dev/null)" || return 2
    [ -n "$c" ] || return 2
    h="$(_code_identity_hash "$c" -C "$root")" || return 2
    printf 'H=%s\n' "$h"
}

# code_identity_same A B [ROOT] -> rc 0 same code, 1 different code, 2 not_measured
code_identity_same() {
    local a b
    a="$(code_identity "${1:-}" "${3:-.}")" || return 2
    b="$(code_identity "${2:-}" "${3:-.}")" || return 2
    [ "$a" = "$b" ] && return 0
    return 1
}

# code_identity_bump_of M B NEWVER -> rc 0 iff B's code is exactly M's after
# `scripts/bump-version.sh NEWVER` (M's own copy of the tool), run in a temporary
# worktree; 1 when B carries anything else; 2 not_measured (unknown rev, the tool
# missing or failing). Leaves the caller's checkout untouched. Wired into no gate.
code_identity_bump_of() {
    local m="${1:-}" b="${2:-}" ver="${3:-}" mc hb wt rc tree hm
    [ -n "$m" ] && [ -n "$b" ] && [ -n "$ver" ] || return 2
    mc="$(git rev-parse --verify --quiet "${m}^{commit}" 2>/dev/null)" || return 2
    hb="$(code_identity "$b")" || return 2
    wt="$(mktemp -d "${TMPDIR:-/tmp}/code-identity-bump.XXXXXX")" || return 2
    rc=2
    if git worktree add --detach --quiet "$wt/w" "$mc" >/dev/null 2>&1; then
        if [ -f "$wt/w/scripts/bump-version.sh" ] &&
            (cd "$wt/w" && bash scripts/bump-version.sh "$ver") >/dev/null 2>&1 &&
            git -C "$wt/w" add -A >/dev/null 2>&1 &&
            tree="$(git -C "$wt/w" write-tree 2>/dev/null)" &&
            hm="$(_code_identity_hash "$tree" -C "$wt/w")"; then
            if [ "H=$hm" = "$hb" ]; then rc=0; else rc=1; fi
        fi
        git worktree remove --force "$wt/w" >/dev/null 2>&1
    fi
    rm -rf -- "${wt:?}"
    return "$rc"
}
