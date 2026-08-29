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
# THE ONE CASE THAT SENTENCE COULD NOT EXPRESS, AND WHY `set-aperture` EXISTS
# --------------------------------------------------------------------------
# PERF-049. `check_no_claim_literals.sh` could not see the claim it was built
# for: RATIO_RE matched ASCII `x` only, so the published `2.93× Ollama` (U+00D7)
# passed, and so did `36.9x over FasterTransformer`, because one intervening
# word defeated the adjacency. Widening the pattern reveals 18 claims that were
# ALREADY IN THE TREE and that the guard had simply been unable to read.
#
# Recording them grows the baseline, and this library refused it — correctly,
# because from the working tree an aperture reveal and a fresh violation are
# the same diff. So the guard could not be widened at all: the ratchet's own
# remedy, "fix the finding instead of recording it", asks a five-whys chain in
# a dated QA archive to describe its own subject matter in euphemism, which is
# the reason `docs/specifications/` is excluded from that guard in the first
# place. The paragraph above names a PR that changes the guard's contract as
# the venue for growth. This is the mechanism that venue needed.
#
# An addition to a `set-aperture` baseline is ADMITTED only if BOTH hold:
#
#   (a) THE LINE PREDATES THE COMPARAND. The entry is `<path>:<line>`, and that
#       line is BYTE-IDENTICAL at the comparand. This is what the working tree
#       cannot answer and the comparand can. It closes PERF-028's laundering
#       shape completely: a claim this branch wrote — or moved, or reflowed —
#       has no byte-identical line at the comparand and is REFUSED, whether it
#       arrived in this commit or five commits back on the same branch.
#
#   (b) THE APERTURE ACTUALLY CHANGED. The owning guard's own source differs
#       between the comparand and the working tree. A PR that does not touch
#       the guard cannot record anything, so "record it instead of fixing it"
#       stays unavailable on every ordinary PR.
#
# Everything else is refused exactly as `set` refuses it, and every admitted
# entry is NAMED in the verdict row. A silent admission would be the defect
# this file is about, one level up.
#
# THE SECOND ENTRY SHAPE, AND WHY (b) DOES NOT GATE IT
# ---------------------------------------------------
# A `<path>:<line>` coordinate is not a property of a claim. PERF-006 added
# `pub mod andon;` near the top of crates/aprender-serve/src/lib.rs, wrote no
# claim at all, and slid three baselined claims onto new coordinates. Under (a)
# every one of them read as a claim this branch WROTE, and there was no legal
# remedy: the new coordinate does not hold that text at the comparand, so it
# could not be re-baselined either. A guard that reds for a reason unrelated to
# its property is a guard people learn to route around.
#
# So `set-aperture` also accepts a CONTENT-KEYED entry:
#
#     <path> TAB <hash16> TAB <count>
#
# admitted when the comparand's copy of <path> carries at least <count> lines
# whose key is <hash16>. The key is computed by the OWNING GUARD, through its
# `--keys-of <file> <display-path>` mode, so there is exactly one definition of
# it; a library re-deriving the hash would agree until the day one side was
# edited, and on that day every entry would read as new.
#
# (b) — "the aperture actually changed" — deliberately does NOT gate this
# shape, and that is the difference that makes it worth having. (b) exists so
# that an ordinary PR cannot record a claim instead of fixing it. Under content
# keys it cannot anyway:
#
#   * a claim this branch WROTE has no line with that key at the comparand, and
#     is refused whether or not the guard moved;
#   * a claim this branch DUPLICATED raises <count> above what the comparand
#     carries, and is refused by the same comparison;
#   * a claim that was already visible to the guard at the comparand is already
#     in the baseline, so there is nothing to add — an addition for it would be
#     a count rise, refused above.
#
# What is left is exactly the two honest cases: a pre-existing claim that MOVED,
# and a pre-existing claim a widened guard can now READ. Requiring (b) for the
# first would put the drift red back, with the same absence of a remedy.
#
# THE RESIDUAL, STATED RATHER THAN LEFT TO BE FOUND. (a) and (b) together do
# not prove the comparand's guard could not ALREADY see the line — proving that
# means running the comparand's guard, and a guard that runs another version of
# itself is a complexity this file will not carry. So a PR that edits the guard
# could also record a pre-existing claim it could already see, instead of
# fixing it. That diff grows a baseline AND edits a guard AND prints every
# admitted coordinate, which is about as loud as an unproven step gets. It is
# strictly narrower than the status quo it replaces, which was that the guard
# could never be widened.
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

# A literal TAB, named so the field splitting below reads as field splitting.
_BR_TAB=$'\t'

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

# `set-aperture`. See the header. Same subset semantics as `_br_cmp_set`, with
# ONE admission: an added `<path>:<line>` whose line is byte-identical at the
# comparand, in a diff that also changes the owning guard.
#
# Both halves fail CLOSED. A coordinate that does not parse, a file the
# comparand does not carry, a line past the end of either copy, an unreadable
# blob, a guard path that was not supplied — every one of them lands in
# BR_DELTA and reds the ratchet. There is no branch here that turns "could not
# check" into "no growth"; that is the shape this whole file exists to refuse.
# How many lines of <ref>:<path> carry key <hash16>, per the OWNING GUARD's own
# `--keys-of`. rc 1 means the question could not be answered — never 0.
#
# The listing is cached per path inside the caller's scratch directory. Without
# it the one-time baseline re-key pays a full keying pass per ENTRY rather than
# per FILE, which on this repository is 137 passes over the same 61 files.
_br_aperture_keycount() { # <root> <ref> <guard> <scratch> <path> <hash16>
    local root="$1" ref="$2" guard="$3" scratch="$4" path="$5" hash="$6"
    local safe cache blob
    safe=$(printf '%s' "$path" | tr -c 'A-Za-z0-9._-' '_')
    cache="$scratch/keys.$safe"
    if [ ! -f "$cache" ]; then
        blob="$scratch/blob.$safe"
        git -C "$root" show "${ref}:${path}" > "$blob" 2>/dev/null || return 1
        if ! bash "$root/$guard" --keys-of "$blob" "$path" > "$cache" 2>/dev/null; then
            rm -f "$cache"
            return 1
        fi
    fi
    # grep -c exits 1 on zero matches, which is an ANSWER (none) and not a
    # failure; it still prints 0. Only the guard failing above is unanswerable.
    grep -cxF "${path}${_BR_TAB}${hash}" "$cache" || true
}

_br_cmp_set_aperture() { # <base-file> <cur-file> <root> <ref> <owning-guard-path>
    local base="$1" cur="$2" root="$3" ref="$4" guard="$5"
    local adds entry path line want got aperture_moved=0 refuse
    local cpath crest chash ccount have scratch

    BR_ADMITTED=""
    BR_DELTA=""
    BR_REMOVED=$(LC_ALL=C comm -23 <(_br_entries "$base") <(_br_entries "$cur") | grep -c . || true)
    adds=$(LC_ALL=C comm -13 <(_br_entries "$base") <(_br_entries "$cur"))
    [ -n "$adds" ] || return 0

    # Scratch for the per-path key cache. If it cannot be made, every
    # content-keyed entry is refused below rather than admitted unchecked.
    scratch=$(mktemp -d 2>/dev/null) || scratch=""

    # (b) is a property of the DIFF, not of an entry, so it is decided once for
    # the whole set. An empty or missing guard path leaves this 0 and every
    # addition is refused: "could not check" is never "no growth".
    if [ -n "$guard" ] && [ -e "$root/$guard" ]; then
        if ! git -C "$root" diff --quiet "$ref" -- "$guard" 2>/dev/null; then
            aperture_moved=1
        fi
    fi

    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        refuse=""
        # A TAB says CONTENT-KEYED. See the header: this shape is not gated on
        # (b), because a moved claim is not an aperture change and requiring a
        # guard edit for it is the drift red with extra steps.
        case "$entry" in
        *"$_BR_TAB"*)
            cpath="${entry%%"$_BR_TAB"*}"
            crest="${entry#*"$_BR_TAB"}"
            chash="${crest%%"$_BR_TAB"*}"
            ccount="${crest#*"$_BR_TAB"}"
            if [ "$ccount" = "$crest" ] || [ -z "$cpath" ]; then
                refuse="not a <path>TAB<hash16>TAB<count> entry"
            fi
            # THE THREE SHAPE CHECKS BELOW ARE DIAGNOSTIC, NOT LOAD-BEARING,
            # and that is measured rather than assumed: deleting either the hex
            # check or the missing-guard check leaves the whole case table
            # GREEN, because a malformed key finds zero lines at the comparand
            # and a missing guard cannot key it, so both are refused one branch
            # later by the count comparison. They stay because a corrupt
            # baseline line would otherwise be reported as "this branch WROTE
            # the claim", pointing a reviewer at innocent code instead of at
            # the entry. The count comparison is what actually refuses; only
            # the zero/non-numeric count check catches a row alone.
            if [ -z "$refuse" ]; then
                case "$chash" in
                    [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) : ;;
                    *) refuse="the key is not 16 lowercase hex digits" ;;
                esac
            fi
            if [ -z "$refuse" ]; then
                case "$ccount" in
                    '' | *[!0-9]* | 0) refuse="the occurrence count is not a positive integer" ;;
                esac
            fi
            if [ -z "$refuse" ] && { [ -z "$guard" ] || [ ! -e "$root/$guard" ]; }; then
                refuse="no owning guard to compute the key with, so the claim cannot be shown to predate the comparand"
            fi
            if [ -z "$refuse" ] && [ -z "$scratch" ]; then
                refuse="no scratch directory, so the comparand could not be keyed"
            fi
            if [ -z "$refuse" ] && ! git -C "$root" cat-file -e "${ref}:${cpath}" 2>/dev/null; then
                refuse="the comparand does not carry $cpath, so the claim cannot predate it"
            fi
            if [ -z "$refuse" ]; then
                if have=$(_br_aperture_keycount "$root" "$ref" "$guard" "$scratch" "$cpath" "$chash"); then
                    if [ "${have:-0}" -lt "$ccount" ]; then
                        refuse="the comparand carries ${have:-0} line(s) with that key, not the $ccount recorded: this branch WROTE or DUPLICATED the claim"
                    fi
                else
                    refuse="$guard could not key ${ref}:${cpath}; \"could not check\" is not \"no growth\""
                fi
            fi
            ;;
        *)
        path="${entry%:*}"
        line="${entry##*:}"
        if [ "$aperture_moved" -ne 1 ]; then
            refuse="the owning guard is unchanged in this diff, so no aperture moved"
        else
            case "$entry" in
                *:*) : ;;
                *)   refuse="not a <path>:<line> coordinate" ;;
            esac
            if [ -z "$refuse" ]; then
                case "$line" in
                    '' | *[!0-9]*) refuse="not a <path>:<line> coordinate" ;;
                esac
            fi
            if [ -z "$refuse" ] && ! git -C "$root" cat-file -e "${ref}:${path}" 2>/dev/null; then
                refuse="the comparand does not carry $path, so the line cannot predate it"
            fi
            if [ -z "$refuse" ]; then
                # No `| head`: an early-exiting reader hands the producer
                # SIGPIPE, and under pipefail that invents a failure. sed reads
                # its whole input.
                want=$(git -C "$root" show "${ref}:${path}" 2>/dev/null | sed -n "${line}p")
                got=$(sed -n "${line}p" "$root/$path" 2>/dev/null)
                if [ -z "$want" ] && [ -z "$got" ]; then
                    refuse="line $line is empty or past the end in BOTH copies"
                elif [ "$want" != "$got" ]; then
                    refuse="this branch WROTE or MOVED that line, so it is a new violation"
                fi
            fi
        fi
            ;;
        esac
        if [ -n "$refuse" ]; then
            BR_DELTA="${BR_DELTA}        + ${entry}
              ${refuse}
"
        else
            BR_ADMITTED="${BR_ADMITTED}        ~ ${entry}
"
        fi
    done <<< "$adds"

    [ -z "$scratch" ] || rm -rf "${scratch:?}"
    BR_DELTA="${BR_DELTA%$'\n'}"
    BR_ADMITTED="${BR_ADMITTED%$'\n'}"
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
    # BOOTSTRAP -- the commit that INTRODUCES a baseline.
    #
    # This library landed AFTER every baseline it ratchets, so no existing one
    # ever met its own first commit. The first new baseline does, and it would
    # be blocked by the very gate it arms: neither protected ref can carry a
    # file that does not exist yet, and ABSENT is a hard failure.
    #
    # Reachable ONLY when all three hold, which is exactly "the file is new":
    #   * the comparand is the real protected ref, not an override. An
    #     overridden ref keeps ABSENT, so the scratch-repo case table still
    #     pins the loud branch;
    #   * neither the merge-base nor the tip of origin/main carries it -- a
    #     branch merely BEHIND a main that already has the baseline resolves
    #     TIP and ratchets normally;
    #   * it is present in the working tree. If it is absent there too, this
    #     is a deletion, and baseline_ratchet_check fails before resolving.
    #
    # It is NOT reachable for any baseline currently in this repository: all of
    # them are on origin/main. This verdict is additive, never a relaxation of
    # a check that passes today.
    if [ "$ref" = "origin/main" ] && [ -f "$root/$path" ]; then
        printf 'BOOTSTRAP\t%s\n' "$ref"
        return 0
    fi
    printf 'ABSENT\t%s\n' "$ref"
    return 0
}

# ---------------------------------------------------------------------------
# The entry point every guard calls.
#
#     baseline_ratchet_check <root> <baseline-path> <set|count|keyed|set-aperture> [<owning-guard-path>]
#
# `set-aperture` takes a fifth argument, the owning guard, and without it every
# addition is refused — see (b) in the header.
#
# rc 0 = the baseline did not grow against a ref this branch cannot rewrite.
# rc 1 = it grew, or growth is UNMEASURABLE. Both are failures, and they are
#        distinguished in the text but never in the status.

baseline_ratchet_check() {
    local root="$1" path="$2" kind="$3" guard="${4:-}"
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
        BOOTSTRAP)
            printf 'REPORT ratchet %s is NOT ARMED on this commit: %s carries\n' "$path" "$ref"
            printf '               no such file, because this commit is the one INTRODUCING\n'
            printf '               it. Its %s entr(ies) are unratcheted for this pull\n' \
                "$(grep -cvE '^[[:space:]]*(#|$)' "$root/$path" 2>/dev/null || printf 0)"
            printf '               request ONLY, and are a REVIEWED claim rather than an\n'
            printf '               enforced one. From the next commit the comparand carries\n'
            printf '               the file and an append is REFUSED.\n'
            return 0 ;;
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
    BR_ADMITTED=""
    # `if` rather than `cmd; rc=$?`: a comparator returns 1 BY DESIGN on the
    # growth path, and a caller running `set -e` (check_no_claim_literals.sh
    # does) would die there before printing a single verdict row -- rc=1 with
    # no evidence, which reads exactly like a broken run. An errexit-safe
    # capture is the difference between a RED and a crash.
    case "$kind" in
        set)   if _br_cmp_set   "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
        count) if _br_cmp_count "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
        keyed) if _br_cmp_keyed "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
        set-aperture)
            if _br_cmp_set_aperture "$base_copy" "$root/$path" "$root" "$ref" "$guard"; then
                cmp_rc=0
            else
                cmp_rc=$?
            fi ;;
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
        if [ -n "$BR_ADMITTED" ]; then
            printf 'ok    ratchet  %s grew by %s APERTURE REVEAL(s) vs %s\n' \
                "$path" "$(printf '%s\n' "$BR_ADMITTED" | grep -c . || true)" \
                "$(git -C "$root" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")"
            printf '%s\n' "$BR_ADMITTED"
            # The guarantee is printed PER SHAPE, because the two shapes prove
            # different things and a row claiming the stronger one for the
            # weaker check is the defect this file is about.
            printf '               each entry above was shown to PREDATE the comparand:\n'
            printf '                 <path>:<line>  the line is BYTE-IDENTICAL there, in a diff\n'
            printf '                                that also changes %s\n' "$guard"
            printf '                 content key    that many lines of the comparand copy of the\n'
            printf '                                file carry it, per %s --keys-of\n' "$guard"
            printf '               They are claims the guard could not READ before, or claims\n'
            printf '               that MOVED. Not claims this branch WROTE. Recorded, not blessed.\n'
        else
            printf 'ok    ratchet  %s did not grow (%s removed) vs %s\n' \
                "$path" "$BR_REMOVED" \
                "$(git -C "$root" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")"
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
