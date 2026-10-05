# shellcheck shell=bash
# lib_code_identity.sh — the ONE definition of "same code" (#4673, BLD-002 row R1).
#
# Contract: contracts/code-identity-v1.yaml. Case table and planted mutants:
# scripts/release/check_code_identity.sh.
#
# The operator ruled the identity is the TREE: every tracked file except the root
# evidence/ path. Recording a result never changes H; any other edit does.
# Train-lead ruling (2026-10-05): evidence/ stays excluded EXCEPT the gate inputs listed
# in CODE_IDENTITY_GATE_INPUTS: files under evidence/ that a gate reads as a threshold,
# a denominator, a ceiling or an accept list, not as a result. They are code.
#
#   H(REV) = sha256 over REV's tracked entries "<mode> <type> <object>\t<path>\0"
#            (git ls-tree -r -z --full-tree order), dropping the root path
#            `evidence` and every path under `evidence/` that is not a gate input.
#   same(A, B)  <=>  H(A) = H(B)
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
#   code_identity_diff A B [ROOT]      -> prints the paths H sees differ; rc 0 or 2
#   code_identity_bump_of M B NEWVER   -> rc 0 iff B is exactly M + bump-version.sh NEWVER
# ROOT is the repository to read (default: the current directory).

# The gate inputs under evidence/. Mirrored, path for path, in contracts/code-identity-v1.yaml
# (gate_inputs); check_code_identity.sh fails when the two differ, and when code reads a
# tracked evidence/ file by name that is neither here nor in its receipt table.
CODE_IDENTITY_GATE_INPUTS=(
    evidence/crux/hf-sources.yaml
    evidence/crux/verb-correspondence.yaml
    evidence/models/supported.yaml
    evidence/parity/EXPECTED_RECEIPTS
    evidence/parity/thresholds.yaml
    evidence/release/context-rungs.json
    evidence/release/surface-ratchet.json
    evidence/verbs/refusals.json
)

# Tracked evidence/ files that code names, and are RESULTS, not gate inputs (bash case
# globs; `*` crosses `/`). Every tracked evidence/ file that crates/, scripts/, src/,
# .github/ or the Makefile names must be a gate input above or match one of these, or
# code_identity_unclassified_reads names it and check_code_identity.sh goes RED. Adding a
# glob here is a reviewed classification: it says the file never decides a verdict.
CODE_IDENTITY_RECEIPT_GLOBS=(
    'evidence/binary/*'              # binary snapshots of a built apr
    'evidence/crux/0.*/*'            # per-release CRUX receipts
    'evidence/dogfood/*'             # per-version, per-host dogfood receipts and verdicts
    'evidence/fleet/*'               # CI run history and test timings (PR tiering, never a release verdict)
    'evidence/github/*'              # GitHub API snapshots
    'evidence/kernels/*'             # per-host kernel receipts
    'evidence/parity/l0-1/*'         # per-host parity records
    'evidence/parity/pin-bump-*'     # pin-bump control runs
    'evidence/parity/props-*'        # server props captures
    'evidence/parity/qwen35/*'       # token dumps
    'evidence/parity/LEDGER.md'      # prose ledger
    'evidence/parity/derived_expiries.json' # DERIVED from the perf matrix, which H already holds
    'evidence/parity-http/*'         # HTTP bench runs
    'evidence/perf*'                 # perf campaign receipts and notes
    'evidence/pr-review/*'           # review backtests
    'evidence/prrev-*'               # review measurements
    'evidence/section-*'             # spec-section dispatch records
    'evidence/ship-*'                # SHIP discharge records
    'evidence/task-*'                # task receipts
    'evidence/tokenizer-parity/*'    # tokenizer parity receipts
    'evidence/gh-*'                  # issue investigation findings
    'evidence/gpu-*'                 # investigation findings
    'evidence/m-gpu-*'               # investigation findings
    'evidence/p2c-*'                 # investigation findings
    'evidence/pmat-*'                # investigation findings
    'evidence/distill-*'             # investigation findings
)

# code_identity_unclassified_reads [ROOT] -> prints each evidence/ file in HEAD that HEAD's code
# names by its exact path and that is neither a gate input nor a receipt; rc 0 (the list
# may be empty), 2 when git fails. Over-inclusive on purpose: a path named only in a
# comment still needs a class.
code_identity_unclassified_reads() {
    local root="${1:-.}" refs tracked p g ok
    tracked="$(git -C "$root" ls-tree -r --name-only --full-tree HEAD -- evidence 2>/dev/null)" || return 2
    refs="$(git -C "$root" grep -hoE '(^|[^A-Za-z0-9_/.-])evidence/[A-Za-z0-9_./-]+' HEAD \
        -- crates scripts src .github Makefile ':!evidence' 2>/dev/null)"
    [ "$?" -le 1 ] || return 2
    refs="$(sed -E 's/^[^e]*//; s/[.]+$//' <<< "$refs" | sort -u)"
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        grep -qxF -- "$p" <<< "$tracked" || continue
        ok=0
        for g in "${CODE_IDENTITY_GATE_INPUTS[@]}"; do
            [ "$p" = "$g" ] && ok=1
        done
        for g in "${CODE_IDENTITY_RECEIPT_GLOBS[@]}"; do
            # shellcheck disable=SC2254,SC2086
            case "$p" in
                $g) ok=1 ;;
            esac
        done
        [ "$ok" = 1 ] || printf '%s\n' "$p"
    done <<< "$refs"
}
# _code_identity_drop_re -> the PCRE that matches an excluded path (on a "<path>" or
# "<meta>\t<path>" record); rc 2 when a listed gate input is not a plain evidence/ path
_code_identity_drop_re() {
    local p alt=""
    for p in "${CODE_IDENTITY_GATE_INPUTS[@]}"; do
        case "$p" in
            evidence/[A-Za-z0-9_]*) ;;
            *) return 2 ;;
        esac
        case "$p" in
            *[!A-Za-z0-9_./-]* | */. | */./* | */.. | */../* | *//* | */) return 2 ;;
        esac
        alt="${alt:+$alt|}${p//./\\.}"
    done
    [ -n "$alt" ] || alt='(?!)'
    printf '%s' "(?!(?:$alt)\\z)evidence(?:/|\\z)"
}

# _code_identity_hash TREEISH [GIT-ARGS...] -> prints the 64-hex H; rc 0 or 2
_code_identity_hash() {
    local treeish="$1" h re
    shift
    re="$(_code_identity_drop_re)" || return 2
    # One entry per NUL-terminated record "<meta>\t<path>"; <meta> holds no tab. grep -z
    # drops the root `evidence` path and everything under `evidence/` but the gate inputs
    # (\z: the whole record, so a path with a trailing newline is not a gate input); grep
    # exits 1 when every entry was dropped, which is still a measurement, and 2 on an error.
    h="$(
        git "$@" ls-tree -r -z --full-tree "$treeish" 2>/dev/null |
            LC_ALL=C grep -zvP "^[^\\t]*\\t$re" |
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

# code_identity_diff A B [ROOT] -> prints, one per line, the paths whose change H sees
# (what a refusal names); rc 0, or 2 when either rev is unknown or git fails
code_identity_diff() {
    local root="${3:-.}" re out
    re="$(_code_identity_drop_re)" || return 2
    git -C "$root" rev-parse --verify --quiet "${1:-}^{commit}" >/dev/null 2>&1 || return 2
    git -C "$root" rev-parse --verify --quiet "${2:-}^{commit}" >/dev/null 2>&1 || return 2
    out="$(
        git -C "$root" -c core.quotePath=true diff --no-renames --name-only "$1" "$2" 2>/dev/null |
            LC_ALL=C grep -vP "^$re"
        st=("${PIPESTATUS[@]}")
        [ "${st[0]}" = 0 ] && [ "${st[1]}" -le 1 ] || exit 2
    )" || return 2
    [ -z "$out" ] || printf '%s\n' "$out"
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
