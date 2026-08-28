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
# CONSEQUENCE, STATED PLAINLY: an entry may only LEAVE a ratcheted baseline
# of kind `set`, `count` or `keyed`. Adding one is not "hard", it is refused,
# and there is no in-branch way around it.
#
# THE ONE GROWTH THAT IS NOT LAUNDERING (kind `coord`)
# ---------------------------------------------------
# There are two different acts and the rule above could not tell them apart:
#
#   LAUNDERING  a baseline entry and its matching violation added in the SAME
#               commit. Nothing detected it before because nothing compared the
#               file to anything; this library exists to refuse it, and it stays
#               refused.
#   WIDENING    the violations were ALREADY in the tree. A better detector now
#               sees them. Nothing new entered.
#
# PERF-010 widened the `[X]`-figure guard from catching 1 of 6 ratio shapes to
# 6 of 6 — it had required the competitor's name to sit immediately after the
# ratio in ASCII, so "36.9x over FasterTransformer" and the U+00D7 form the
# epic's own headline fabrication was written in both walked past it. Fourteen
# pre-existing claims surfaced. Every one of them was in the tree at the
# comparand, verbatim; three had merely MOVED because the same commit inserted
# 28 lines above them. This ratchet refused all fourteen, and the only ways out
# were to delete eight dated `docs/qa/` post-mortems — seven of which are
# admissions that apr is SLOWER, and under-claiming is as much a reporting
# failure here as over-claiming — or to drop the detector improvement. Neither
# is a thing a bookkeeping rule should be able to force.
#
# So kind `coord` adds a rule that is MECHANICAL, never an argument in a header:
#
#     growth is permitted only when every added entry's TEXT is already present
#     at the comparand, in the same file, at no greater number of occurrences.
#
#   * TEXT, not coordinate. A line that moved is not a new claim, and a
#     coordinate-keyed test calls it one — the same lesson that re-keyed
#     check_no_claim_literals.sh off FILE:LINE.
#   * PER ENTRY, not per batch. One added entry whose text is absent at the
#     comparand fails the whole file, however many of its neighbours qualify.
#   * OCCURRENCES MAY NOT RISE. Otherwise a launderer could copy an existing
#     baselined claim to a second place in the same file and call it a widening.
#   * FAIL CLOSED. An entry that is not a `<path>:<line>` coordinate, a path
#     absent at the comparand, a line past end of file, an empty line, an
#     unreadable blob: none of these ESTABLISHES a widening, so each one leaves
#     the growth refused. "Could not tell" is never green.
#   * OPT IN. `set` is unchanged and stays the default. A baseline gets `coord`
#     only when its entries really are coordinates AND a detector widening is
#     the growth it must be able to express. Today that is
#     claim_literal_baseline.txt alone; perf_claim_citation_baseline.txt is
#     coordinate-keyed too and is deliberately left on the stricter `set`,
#     because nothing has needed it.
#
# The row this prints says GREW-BY-WIDENING, never "did not grow". A gate that
# reports no growth after growth is the lying-verdict class this whole file is
# about.
#
# KNOWN LIMIT, stated rather than discovered later: `coord` cannot tell a
# widening from a commit that MOVES a pre-existing violation to a new line and
# baselines it there. Both leave the text present at the comparand at the same
# count. The claim is the same claim either way, so the guard's own subject
# matter is unchanged — but it is a hole, and it is here in writing.
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

# ---------------------------------------------------------------------------
# _br_cmp_coord — subset semantics as `set`, PLUS the widening rule above.
#
# Unlike the other comparators this one needs the repository, not just two
# files: establishing "the text was already there" means reading <path> out of
# the comparand COMMIT, which is a different blob per entry.
_br_cmp_coord() { # _br_cmp_coord <base-file> <cur-file> <root> <ref>
    local base="$1" cur="$2" root="$3" ref="$4"
    local added entry path lineno text n_now n_base blob slug tmp
    BR_DELTA=""
    BR_WIDENED=0
    BR_REMOVED=$(LC_ALL=C comm -23 <(_br_entries "$base") <(_br_entries "$cur") | grep -c . || true)
    added=$(LC_ALL=C comm -13 <(_br_entries "$base") <(_br_entries "$cur"))
    [ -n "$added" ] || return 0

    tmp=$(mktemp -d) || {
        BR_DELTA='        + no scratch directory, so no widening can be ESTABLISHED and the
          growth stands refused. "Could not tell" is not green.'
        return 1
    }

    # A heredoc, never `printf | while`: the loop must run in THIS shell or
    # every BR_* assignment below is discarded with the subshell.
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        path=${entry%:*}
        lineno=${entry##*:}
        case "$lineno" in
            '' | *[!0-9]*)
                BR_DELTA="${BR_DELTA}"$'\n'"        + ${entry}  -- not a <path>:<line> coordinate"
                continue ;;
        esac
        if [ ! -f "$root/$path" ]; then
            BR_DELTA="${BR_DELTA}"$'\n'"        + ${entry}  -- no such file in the working tree"
            continue
        fi
        text=$(sed -n "${lineno}p" "$root/$path")
        if [ -z "$text" ]; then
            BR_DELTA="${BR_DELTA}"$'\n'"        + ${entry}  -- that line is empty or past end of file"
            continue
        fi
        slug=$(printf '%s' "$path" | tr -c 'A-Za-z0-9' '_')
        blob="$tmp/$slug"
        if [ ! -f "$blob" ] && ! git -C "$root" show "${ref}:${path}" > "$blob" 2>/dev/null; then
            rm -f "$blob"
            BR_DELTA="${BR_DELTA}"$'\n'"        + ${entry}  -- ${path} does not exist at the comparand"
            continue
        fi
        n_base=$(LC_ALL=C grep -Fxc -- "$text" "$blob" 2>/dev/null) || n_base=0
        n_now=$(LC_ALL=C grep -Fxc -- "$text" "$root/$path" 2>/dev/null) || n_now=0
        if [ "$n_base" -gt 0 ] && [ "$n_now" -le "$n_base" ]; then
            BR_WIDENED=$((BR_WIDENED + 1))
            continue
        fi
        if [ "$n_base" -eq 0 ]; then
            BR_DELTA="${BR_DELTA}"$'\n'"        + ${entry}  -- NEW TEXT. Absent from ${path} at the comparand:"
        else
            BR_DELTA="${BR_DELTA}"$'\n'"        + ${entry}  -- that text now occurs ${n_now}x, was ${n_base}x at the comparand:"
        fi
        BR_DELTA="${BR_DELTA}"$'\n'"              ${text:0:100}"
    done <<EOF
$added
EOF

    rm -rf "${tmp:?}"
    BR_DELTA=${BR_DELTA#$'\n'}
    [ -z "$BR_DELTA" ]
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
#     baseline_ratchet_check <root> <baseline-path-relative-to-root> <set|coord|count|keyed>
#
# rc 0 = the baseline did not grow against a ref this branch cannot rewrite.
# rc 1 = it grew, or growth is UNMEASURABLE. Both are failures, and they are
#        distinguished in the text but never in the status.

baseline_ratchet_check() {
    local root="$1" path="$2" kind="$3"
    local resolution mode ref tmp base_copy how note cmp_rc short

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
    BR_WIDENED=0
    # `if` rather than `cmd; rc=$?`: a comparator returns 1 BY DESIGN on the
    # growth path, and a caller running `set -e` (check_no_claim_literals.sh
    # does) would die there before printing a single verdict row -- rc=1 with
    # no evidence, which reads exactly like a broken run. An errexit-safe
    # capture is the difference between a RED and a crash.
    case "$kind" in
        set)   if _br_cmp_set   "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
        coord) if _br_cmp_coord "$base_copy" "$root/$path" "$root" "$ref"; then cmp_rc=0; else cmp_rc=$?; fi ;;
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
        short=$(git -C "$root" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")
        if [ "${BR_WIDENED:-0}" -gt 0 ]; then
            # NOT "did not grow". It grew. Say so, and say what made it legal.
            printf 'ok    widened  %s GREW BY %s, and every added entry is a\n' "$path" "$BR_WIDENED"
            printf '               WIDENING: its text is already in the same file at %s, at no\n' "$short"
            printf '               greater count. Nothing new entered the tree -- a better\n'
            printf '               detector saw what was already there. (%s removed.)\n' "$BR_REMOVED"
        else
            printf 'ok    ratchet  %s did not grow (%s removed) vs %s\n' \
                "$path" "$BR_REMOVED" "$short"
        fi
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
