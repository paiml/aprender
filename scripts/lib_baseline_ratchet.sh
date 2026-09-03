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
#   (a) THE LINE PREDATES THE COMPARAND, by either of two proofs. The entry is
#       `<path>:<line>`, and EITHER
#         (a1) that line is BYTE-IDENTICAL at the comparand, OR
#         (a2) it MOVED: the working tree's text for that line occurs in the
#              SAME FILE at the comparand, at no greater number of occurrences.
#       This is what the working tree cannot answer and the comparand can. It
#       closes PERF-028's laundering shape: a claim this branch wrote has no
#       text at the comparand at all and is REFUSED, whether it arrived in this
#       commit or five commits back on the same branch.
#
#       (a2) IS NOT A RELAXATION FOR CONVENIENCE; WITHOUT IT THE RULE IS A TRAP.
#       (a1) alone keys the admission on a COORDINATE, and a coordinate is not
#       the claim — the text is. PERF-019 inserted §3.3.1, twenty-eight lines,
#       into docs/benchmarking-gate-spec.md at line 162. Two claims already
#       baselined at :235 and :308 — the spec quoting the fabricated `2.93×
#       Ollama` in order to BAN it — slid to :263 and :336 untouched. Under
#       (a1) those are two brand-new violations and the old entries go stale,
#       so the only legal moves are to delete a document that exists to record
#       the fabrication, or to abandon an unrelated subsection. Neither is a
#       thing a bookkeeping rule should be able to force, and the trap arms for
#       ANY edit above ANY baselined line in a file this guard reads.
#
#       The counting half is what keeps (a2) from being a hole: a launderer who
#       COPIES an already-baselined claim to a second site in the same file
#       raises its occurrence count, and that is refused. Text absent at the
#       comparand is refused. Only a genuine relocation passes, and the claim
#       is the same claim either way, so the guard's subject matter is
#       unchanged. Both halves are rows in check_baseline_ratchets.sh.
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
# ONE admission: an added `<path>:<line>` whose line PREDATES the comparand --
# byte-identical there at the same coordinate, or, when it has MOVED, present
# in the same file at the comparand at no greater number of occurrences -- in a
# diff that also changes the owning guard.
#
# Both halves fail CLOSED. A coordinate that does not parse, a file the
# comparand does not carry, a line past the end of either copy, an unreadable
# blob, a guard path that was not supplied — every one of them lands in
# BR_DELTA and reds the ratchet. There is no branch here that turns "could not
# check" into "no growth"; that is the shape this whole file exists to refuse.
_br_cmp_set_aperture() { # <base-file> <cur-file> <root> <ref> <owning-guard-path>
    local base="$1" cur="$2" root="$3" ref="$4" guard="$5"
    local adds entry path line want got aperture_moved=0 refuse n_base n_now

    BR_ADMITTED=""
    BR_DELTA=""
    BR_REMOVED=$(LC_ALL=C comm -23 <(_br_entries "$base") <(_br_entries "$cur") | grep -c . || true)
    adds=$(LC_ALL=C comm -13 <(_br_entries "$base") <(_br_entries "$cur"))
    [ -n "$adds" ] || return 0

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
                    # (a2), the MOVE. See the header. The coordinate is not the
                    # claim; the TEXT is. `grep -c` reads its whole input, so
                    # there is no early exit to hand anyone a SIGPIPE, and it
                    # exits 1 on a zero count -- hence the `|| n=0`.
                    n_base=$(git -C "$root" show "${ref}:${path}" 2>/dev/null \
                             | LC_ALL=C grep -Fxc -- "$got") || n_base=0
                    n_now=$(LC_ALL=C grep -Fxc -- "$got" "$root/$path" 2>/dev/null) || n_now=0
                    if [ "$n_base" -eq 0 ]; then
                        refuse="this branch WROTE that line: its text is absent from $path at the comparand"
                    elif [ "$n_now" -gt "$n_base" ]; then
                        refuse="that text now occurs ${n_now}x in $path and was ${n_base}x at the comparand, so this branch ADDED one"
                    fi
                fi
            fi
        fi
        if [ -n "$refuse" ]; then
            BR_DELTA="${BR_DELTA}        + ${entry}
              ${refuse}
"
        else
            BR_ADMITTED="${BR_ADMITTED}        ~ ${entry}
"
        fi
    done <<< "$adds"

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

# `keyed2`. The same rule as `keyed` over lines carrying TWO integers,
# `<key> <a> <b>`, whitespace-separated. It exists because the complexity
# ratchet records a pair per function -- cyclomatic AND cognitive -- and the
# rule is "over EITHER", so a baseline holding only one of them would ratchet
# only half the predicate while looking complete.
#
# Splitting the pair into two `keyed` rows was the alternative and it is worse:
# the key would have to carry the metric name, and a reader of
# scripts/complexity_baseline.txt could no longer see, on one line, what a
# function costs.
#
# No key may appear and neither number may rise. Either number may FALL, which
# is what makes a partial refactor recordable rather than a diff the meta-gate
# refuses. Comment stripping happens in grep, not in awk: an awk program
# carrying a bracket-and-paren regex reads to bashrs as a `[ ` test and lands
# SC1028 error lines in a shrink-only lint baseline.
_br_cmp_keyed2() { # _br_cmp_keyed2 <base-file> <cur-file>  (lines are <key> <int> <int>)
    BR_DELTA=$(LC_ALL=C awk '
        NR == FNR { a[$1] = $2; b[$1] = $3; seen[$1] = 1; next }
        {
            if (!($1 in seen))    { printf "        + NEW KEY  %s (%s %s)\n", $1, $2, $3 }
            else {
                if ($2+0 > a[$1]+0) { printf "        + RAISED   %s  %s -> %s\n", $1, a[$1], $2 }
                if ($3+0 > b[$1]+0) { printf "        + RAISED   %s  %s -> %s\n", $1, b[$1], $3 }
            }
        }
    ' <(_br_data "$1") <(_br_data "$2"))
    BR_REMOVED=$(LC_ALL=C awk '
        NR == FNR { a[$1] = $2; b[$1] = $3; seen[$1] = 1; next }
        { if (!($1 in seen) || a[$1]+0 < $2+0 || b[$1]+0 < $3+0) { n++ } }
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
#     baseline_ratchet_check <root> <baseline-path> <set|count|keyed|keyed2|set-aperture> [<owning-guard-path>]
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
        keyed2) if _br_cmp_keyed2 "$base_copy" "$root/$path"; then cmp_rc=0; else cmp_rc=$?; fi ;;
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
            printf '               each line above PREDATES the comparand -- byte-identical there,\n'
            printf '               or moved with its text intact and no more occurrences than before\n'
            printf '               -- and this diff changes %s. They are claims the\n' "$guard"
            printf '               guard could not READ before, not claims this branch WROTE.\n'
            printf '               Recorded, not blessed.\n'
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
