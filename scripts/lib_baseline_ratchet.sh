#!/usr/bin/env bash
# lib_baseline_ratchet.sh — turn a "shrink-only" baseline into an actual ratchet.
#
# THE DEFECT THIS EXISTS FOR
# --------------------------
# check_no_fabricated_baselines.sh printed
#
#     ok  rust  37 ledgered site(s) = 37 ledger entr(ies), 0 new, 0 stale
#               (shrink-only, PERF-008-RUST; the count is enforced, not asserted)
#
# while appending one line to its ledger and landing a fabrication at exactly
# that coordinate returned rc=0. The words "SHRINK-ONLY" appeared four times in
# that file; a comparison against anything appeared zero times.
#
# The transferable diagnosis, and the reason this is a library rather than one
# more fix: NEW and STALE are the two properties checkable from the WORKING
# TREE, and neither one is a ratchet.
#
#   * NEW   — a finding with no baseline entry.  A ledgered finding is not new.
#   * STALE — a baseline entry with no finding.   A real finding is not stale.
#
# A baseline line and its matching violation, added in the SAME commit, satisfy
# both at once. That commit is the laundering shape, and every guard in this
# repository that called itself shrink-only accepted it: a sweep appending one
# entry cloned from each file's own last real entry found 12 of 12 green.
#
# A ratchet is a property of the DIFF against a ref the author cannot rewrite.
# Nothing derivable from the working tree alone can be one, because the working
# tree contains both the rule and the exception to it.
#
# WHY A SUBSET AND NOT A COUNT
# ----------------------------
# Counting passes a SWAP: drop one coordinate, add another, total unchanged.
# That is an append wearing the old total. The current entry set must be a
# SUBSET of the comparand's — removal is the point of a ratchet and stays
# green, and any entry not already on the comparand fails whether it arrived by
# append or by substitution.
#
# For the baselines that hold a single integer rather than a list, the integer
# itself is the whole state, and "current <= comparand" is the same property
# with a one-element universe. For `path<TAB>count` ledgers the property is
# per-key: no key may rise and no key may appear.
#
# THE COMPARAND IS A REF A PULL REQUEST CANNOT REWRITE
# ----------------------------------------------------
# This mirrors check_dogfood_coverage.sh, which exists because a floor and its
# universe both lived in one editable file: "There is no baseline NUMBER in
# this repository for a PR to edit."
#
#   * merge-base(HEAD, origin/main) is PREFERRED — it isolates this branch's
#     own edits, so a branch merely behind main stays green. It needs shared
#     history.
#   * the origin/main TIP is the FALLBACK, and it is not decoration: CI checks
#     this repository out at fetch-depth 1, so a grafted shallow head has no
#     common ancestor and the CI path IS the tip path. The tip is strictly
#     stronger (it also forbids re-adding an entry main has already deleted).
#     The cost is a false red on a branch behind a main that already shrank the
#     baseline; the remedy is `git rebase origin/main`, and the FAIL says so.
#   * if NEITHER resolves, this is a HARD FAILURE. It never degrades to
#     comparing the branch against itself, which would disarm every ratchet
#     silently — the exact failure this library is about.
#   * if the ref resolves but carries no baseline, that is ABSENT and also a
#     hard failure. A missing measurement is never "no growth".
#
# Set BASELINE_RATCHET_BASE_REF to override the comparand. Every row then says
# the ref is NOT protected, because a gate that keeps printing its guarantee
# after the guarantee was overridden is lying in the way this whole file is
# about.
#
# CONSEQUENCE, STATED PLAINLY: an entry may only LEAVE a ratcheted baseline.
# Adding one is not "hard", it is refused, and there is no in-branch way around
# it. If a baseline genuinely must grow, that is a decision to argue in a PR
# that changes the guard's contract — not a line to slip into a data file.
#
# OPTION-NEUTRAL. This file is SOURCED, and `set` in a sourced file mutates the
# CALLER's shell (see check_sourced_libs_option_neutral.sh, and the nightly it
# killed six lines in). There is no `set` at file scope here; every entry point
# reports by RETURN STATUS. Source it as:
#
#     . "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
#
# Refs: paiml/aprender#2706 (APR-PERF-GATE-001), PERF-008, PERF-028.

BASELINE_RATCHET_BASE_REF="${BASELINE_RATCHET_BASE_REF:-origin/main}"

# ---------------------------------------------------------------------------
# Readers. Deliberately no pipe whose READER can exit early: `grep … | head -1`
# returns 141 under pipefail when the writer takes SIGPIPE, which is
# input-size dependent and therefore green locally and red in CI at random.
# `sort` and `comm` consume their whole input, so those pipes are safe.

_br_data() { # _br_data <file>  -> the data lines, in file order
    grep -vE '^[[:space:]]*(#|$)' "$1" 2>/dev/null || true
}

_br_entries() { # _br_entries <file>  -> data lines, sorted, deduplicated
    _br_data "$1" | LC_ALL=C sort -u
}

_br_number() { # _br_number <file> -> the single integer it holds, rc 1 if it holds none
    local all first
    all=$(_br_data "$1")
    first=${all%%$'\n'*}
    first=${first//[[:space:]]/}
    case "$first" in
        '' | *[!0-9]*) return 1 ;;
    esac
    printf '%s\n' "$first"
}

# ---------------------------------------------------------------------------
# Comparators. Each sets BR_DELTA to the human-readable growth it refused and
# BR_REMOVED to the number of entries that legitimately left.

_br_cmp_set() { # _br_cmp_set <base-file> <cur-file>
    BR_DELTA=$(LC_ALL=C comm -13 <(_br_entries "$1") <(_br_entries "$2") | sed 's/^/        + /')
    BR_REMOVED=$(LC_ALL=C comm -23 <(_br_entries "$1") <(_br_entries "$2") | grep -c . || true)
    [ -z "$BR_DELTA" ]
}

_br_cmp_count() { # _br_cmp_count <base-file> <cur-file>
    local b c
    BR_DELTA=""
    BR_REMOVED=0
    b=$(_br_number "$1") || { BR_DELTA="        comparand holds no integer"; return 2; }
    c=$(_br_number "$2") || { BR_DELTA="        working tree holds no integer"; return 2; }
    if [ "$c" -gt "$b" ]; then
        BR_DELTA=$(printf '        + the recorded count rose %s -> %s' "$b" "$c")
        return 1
    fi
    if [ "$c" -lt "$b" ]; then
        BR_REMOVED=$((b - c))
    fi
    return 0
}

# Comment stripping happens in grep, NOT in awk. An awk program carrying
# `/^[ \t]*#/` reads to a shell linter as a `[ ` test with parentheses inside
# it: bashrs reports SC1028/SC2104 errors against a line that is awk source,
# and check_shell_lint_ratchet.sh counts error LINES, so a false positive still
# moves a shrink-only baseline. Feeding awk pre-filtered data keeps both the
# awk simpler and the lint honest.
_br_cmp_keyed() { # _br_cmp_keyed <base-file> <cur-file>   (lines are <key><TAB><integer>)
    BR_DELTA=$(LC_ALL=C awk -F'\t' '
        NR == FNR { b[$1] = $2; seen[$1] = 1; next }
        {
            if (!($1 in seen))       { printf "        + NEW KEY  %s (%s)\n", $1, $2 }
            else if ($2+0 > b[$1]+0) { printf "        + RAISED   %s  %s -> %s\n", $1, b[$1], $2 }
        }
    ' <(_br_data "$1") <(_br_data "$2"))
    BR_REMOVED=$(LC_ALL=C awk -F'\t' '
        NR == FNR { c[$1] = $2; next }
        { if (!($1 in c) || c[$1]+0 < $2+0) { n++ } }
        END { print n+0 }
    ' <(_br_data "$2") <(_br_data "$1"))
    [ -z "$BR_DELTA" ]
}

# ---------------------------------------------------------------------------
# Comparand resolution. Returns "<MODE>\t<commit-ish>" and never fails: the
# CALLER decides, so that "could not resolve" is a loud verdict row rather than
# a swallowed error.

baseline_ratchet_resolve() { # baseline_ratchet_resolve <root> <ref> <path>
    local root="$1" ref="$2" path="$3" mb
    if ! git -C "$root" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null 2>&1; then
        printf 'UNRESOLVABLE\t%s\n' "$ref"
        return 0
    fi
    mb=$(git -C "$root" merge-base HEAD "$ref" 2>/dev/null) || mb=""
    if [ -n "$mb" ] && git -C "$root" cat-file -e "${mb}:${path}" 2>/dev/null; then
        printf 'MERGEBASE\t%s\n' "$mb"
        return 0
    fi
    if git -C "$root" cat-file -e "${ref}:${path}" 2>/dev/null; then
        printf 'TIP\t%s\n' "$ref"
        return 0
    fi
    printf 'ABSENT\t%s\n' "$ref"
    return 0
}

# ---------------------------------------------------------------------------
# The entry point every guard calls.
#
#     baseline_ratchet_check <root> <baseline-path-relative-to-root> <set|count|keyed>
#
# rc 0 = the baseline did not grow against a ref this branch cannot rewrite.
# rc 1 = it grew, or growth is UNMEASURABLE. Both are failures, and they are
#        distinguished in the text but never in the status.

baseline_ratchet_check() {
    local root="$1" path="$2" kind="$3"
    local resolution mode ref tmp base_copy how note cmp_rc

    if [ ! -f "$root/$path" ]; then
        printf 'FAIL  ratchet  %s is missing from the working tree. Growth is\n' "$path"
        printf '               UNMEASURED without it, and an unmeasured ratchet is not a\n'
        printf '               ratchet. Restore it, or retire the check in the same commit.\n'
        return 1
    fi

    resolution=$(baseline_ratchet_resolve "$root" "$BASELINE_RATCHET_BASE_REF" "$path")
    mode=${resolution%%$'\t'*}
    ref=${resolution##*$'\t'}

    case "$mode" in
        UNRESOLVABLE)
            printf 'FAIL  ratchet  cannot resolve the comparand ref <%s>, so shrink-only\n' "$ref"
            printf '               for %s is UNMEASURED. It is NOT degraded to\n' "$path"
            printf '               comparing this branch against itself — that disarms the\n'
            printf '               ratchet silently. In CI, before this guard runs:\n'
            printf '               git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main\n'
            return 1 ;;
        ABSENT)
            printf 'FAIL  ratchet  %s carries no %s, so there is nothing\n' "$ref" "$path"
            printf '               to shrink from. A missing comparand is not "no growth".\n'
            printf '               Either this branch predates the baseline (git rebase\n'
            printf '               origin/main), or the baseline was deleted to escape its\n'
            printf '               own gate. Retire the check in the same commit if it is\n'
            printf '               genuinely being retired.\n'
            return 1 ;;
    esac

    tmp=$(mktemp -d) || {
        printf 'FAIL  ratchet  could not create a scratch directory, so %s is\n' "$path"
        printf '               UNMEASURED. That is a failure, not a skip.\n'
        return 1
    }
    base_copy="$tmp/base"
    if ! git -C "$root" show "${ref}:${path}" > "$base_copy" 2>/dev/null; then
        rm -rf "${tmp:?}"
        printf 'FAIL  ratchet  could not read %s:%s\n' "$ref" "$path"
        return 1
    fi

    BR_DELTA=""
    BR_REMOVED=0
    # `if` rather than `cmd; rc=$?`: a comparator returns 1 BY DESIGN on the
    # growth path, and a caller running `set -e` (check_no_claim_literals.sh
    # does) would die there before printing a single verdict row -- rc=1 with
    # no evidence, which reads exactly like a broken run. An errexit-safe
    # capture is the difference between a RED and a crash.
    case "$kind" in
        set)   if _br_cmp_set   "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
        count) if _br_cmp_count "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
        keyed) if _br_cmp_keyed "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
        *)
            rm -rf "${tmp:?}"
            printf 'FAIL  ratchet  unknown comparison kind <%s> for %s.\n' "$kind" "$path"
            return 1 ;;
    esac
    rm -rf "${tmp:?}"

    case "$mode" in
        MERGEBASE) how="merge-base with $BASELINE_RATCHET_BASE_REF" ;;
        TIP)       how="tip of $BASELINE_RATCHET_BASE_REF (no merge-base available; stricter)" ;;
        *)         how="$mode" ;;
    esac
    note="protected; a pull request cannot rewrite it"
    if [ "$BASELINE_RATCHET_BASE_REF" != "origin/main" ]; then
        note="OVERRIDDEN via BASELINE_RATCHET_BASE_REF — NOT a protected ref"
    fi

    if [ "$cmp_rc" -eq 0 ]; then
        printf 'ok    ratchet  %s did not grow (%s removed) vs %s\n' \
            "$path" "$BR_REMOVED" \
            "$(git -C "$root" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")"
        printf '               comparand: %s (%s)\n' "$how" "$note"
        return 0
    fi

    if [ "$cmp_rc" -eq 2 ]; then
        printf 'FAIL  ratchet  %s could not be compared:\n' "$path"
    else
        printf 'FAIL  ratchet  %s GREW. It is SHRINK-ONLY:\n' "$path"
    fi
    printf '%s\n' "$BR_DELTA"
    printf '               comparand: %s (%s)\n' "$how" "$note"
    printf '               An entry may only LEAVE this file. Fix the finding instead of\n'
    printf '               recording it. If the branch is merely behind a main that has\n'
    printf '               already shrunk this baseline: git rebase origin/main.\n'
    return 1
}
