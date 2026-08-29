#!/usr/bin/env bash
# check_mutation_registry.sh — APR-PERF-GATE-001 §5 must describe the tree.
#
# WHY THIS EXISTS (PERF-047, aprender#2752)
# -----------------------------------------
# §5 is the epic's own ledger: "a gate with no registered mutation is
# inadmissible", so which arms may contribute to §4.8's verdict is decided by
# what §5 says. Nothing compared it to the repository, and by 2026-08-29 it was
# wrong in BOTH directions at once:
#
#   recorded as unproven, actually proven   12 rows
#   recorded as proof, not proven            2 rows
#   rows the registry did not have           1 (cell completeness)
#
# Three of the twelve said `**not written**` beside guards that ship case
# tables and are each invoked TWICE in ci.yml. The two in the other direction
# are the dangerous ones: `2.93x Ollama` spelled with U+00D7 — the mutation the
# claim-literal row named — leaves that guard GREEN, so a row read as evidence
# pointed at an input the gate cannot fail on.
#
# THE CHECK IS TWO-WAY, because the drift was.
#
#   R0  the §5 table parses and has at least MIN_ROWS rows.
#   R1  every Status cell opens with PROVEN | PARTIAL | UNCOVERED | UNPROVEN.
#   R2  a PROVEN/PARTIAL/UNCOVERED row names a backticked file that EXISTS.
#   R3  the Mutation and Discrimination cells are non-empty.
#   R4  an UNPROVEN row may not name a file whose self-test a workflow runs
#       (say UNCOVERED and quote the mutation that stays green), and an
#       UNCOVERED row must name one (else it is a softer spelling of UNPROVEN).
#   R5  the set of not-PROVEN rows is SHRINK-ONLY against §5 AT A REF THIS
#       BRANCH CANNOT REWRITE. There is no separate baseline file: the registry
#       is its own baseline, so there is nowhere to record a downgrade.
#
# THE FIRST MIGRATION IS WEAKER THAN EVERY LATER RUN, AND SAYS SO. At the
# comparand commit (31732f5db) every §5 status is still prose, so the scanner
# reads all 24 rows as INVALID and therefore not-PROVEN. `current ⊆ base` is
# then satisfied by anything, and R5 cannot bite on the commit that introduces
# it — measured: downgrading a row from PROVEN to PARTIAL here returns rc=0.
# That is the correct semantics (a registry with no usable vocabulary proves
# nothing), not a disarm, and it is stated because a property that only starts
# holding later is exactly the kind that never gets checked. Once this lands,
# main carries the vocabulary, the base set is the ten not-PROVEN rows, and the
# same downgrade is refused. Demonstrated on the landing commit with
# BASELINE_RATCHET_BASE_REF pointed at it, which prints its own NOT-protected
# warning.
#
# WHAT IT DELIBERATELY DOES NOT CLAIM. It cannot read an English Mutation cell
# and decide the sentence describes something real — the U+00D7 row is exactly
# that failure and was caught by RUNNING the mutation, not by any parser. The
# counts in a PROVEN cell are a human claim; the standing rule is that a PR
# moving a row to PROVEN quotes the RED, the discrimination and the revert.
#
#   bash scripts/check_mutation_registry.sh              # check
#   bash scripts/check_mutation_registry.sh --self-test  # case table
#
# Refs: paiml/aprender#2706 (APR-PERF-GATE-001), #2752 (PERF-047).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCANNER="${REPO_ROOT}/scripts/lib/mutation_registry.py"

# Vacuity floor. §5 held 24 rows at v2.2 and 25 after PERF-047; a parser that
# silently stopped matching would report zero rows, zero violations and look
# like health — the exact failure this whole epic is about.
MIN_ROWS="${MIN_ROWS:-20}"

# `scan <root>` -> the scanner's TSV on stdout, non-zero if the scan broke.
# Callers capture to a FILE rather than piping: a `producer | grep -q` pair
# hands the pipeline the producer's SIGPIPE status and invents a failure, which
# has cost this repository four separate false verdicts.
scan() {
    python3 "$SCANNER" "$1"
}

# `grep -c` always PRINTS a count and returns 1 when that count is zero, so an
# `|| printf 0` fallback appends a SECOND number. Measured while writing this:
# the vacuity comparison died with `[: 0\n0: integer expression expected` -- a
# guard crashing where it meant to compare. Every count below therefore ends in
# `|| true`, which keeps the printed count and discards only the status.

rows_of() { # rows_of <tsv> <status|ANY> -> matching row keys, sorted
    LC_ALL=C awk -F'\t' -v want="$2" \
        '$1 == "ROW" && (want == "ANY" || $2 == want) { print $3 }' "$1" \
        | LC_ALL=C sort -u
}

not_proven_rows() { # not_proven_rows <tsv> -> row keys that are not PROVEN
    LC_ALL=C awk -F'\t' '$1 == "ROW" && $2 != "PROVEN" { print $3 }' "$1" \
        | LC_ALL=C sort -u
}

render_violations() {
    LC_ALL=C awk -F'\t' '$1 == "VIOLATION" {
        printf "  %-4s %-32.32s %s\n", $2, $3, $4
    }' "$1"
}

# ---------------------------------------------------------------- selftest --
# Every row states the mutation it makes to a synthetic tree and which rule must
# fire. Row 1 is the control: without it, a checker that flagged EVERYTHING
# would pass every other row.
selftest() {
    local td pass=0 fail=0
    td="$(mktemp -d)" || { printf 'FAIL  mktemp -d failed\n'; return 2; }
    case "$td" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'FAIL  mktemp -d gave %s, refusing to rm -rf it\n' "${td:-<empty>}"
           return 2 ;;
    esac

    mk_root() { # mk_root <name> <table-row> [<workflow steps, %b-escaped>]
        local r="$td/$1"
        rm -rf "${r:?}"
        mkdir -p "$r/docs/specifications" "$r/.github/workflows" "$r/scripts"
        printf '#!/usr/bin/env bash\ncase "${1:-}" in\n  --selftest) : ;;\nesac\n' \
            > "$r/scripts/real_guard.sh"
        {
            printf '# APR-PERF-GATE-001 vT\n\n## §5 Mutation registry\n\n'
            printf '| Gate / control | File `[C]` | Mutation → RED | Discrimination | Status |\n'
            printf '|---|---|---|---|---|\n'
            printf '%s\n' "$2"
            printf '\n## §6 Next\n'
        } > "$r/docs/specifications/APR-PERF-GATE-001-vT.md"
        {
            printf 'jobs:\n  x:\n    steps:\n'
            printf '%b' "${3:-      - run: bash scripts/real_guard.sh --selftest\\n}"
        } > "$r/.github/workflows/ci.yml"
        printf '%s' "$r"
    }

    row() { # row <name> <expected rules, comma-separated, or CLEAN> <root>
        local name="$1" want="$2" root="$3" got
        if ! scan "$root" > "$td/out.tsv" 2>"$td/err.txt"; then
            printf '  BROKE %-46s scanner errored: %s\n' "$name" "$(head -1 "$td/err.txt")"
            fail=$((fail + 1)); return
        fi
        got=$(LC_ALL=C awk -F'\t' '$1 == "VIOLATION" { print $2 }' "$td/out.tsv" \
              | LC_ALL=C sort -u | tr '\n' ',' | sed 's/,$//')
        [ -n "$got" ] || got=CLEAN
        if [ "$got" = "$want" ]; then
            printf '  ok    %-46s %s\n' "$name" "$want"; pass=$((pass + 1))
        else
            printf '  BROKE %-46s expected %s got %s\n' "$name" "$want" "$got"
            fail=$((fail + 1))
        fi
    }

    local BT GOOD
    BT=$(printf '\140')          # a literal backtick, built rather than typed
    GOOD="| Real gate | ${BT}scripts/real_guard.sh${BT} | flip the floor | at-floor green | PROVEN - 9/9 |"

    row "a clean registry is silent" CLEAN "$(mk_root clean "$GOOD")"

    local UNP_ROW UNC_ROW
    UNP_ROW="| Real gate | ${BT}scripts/real_guard.sh${BT} | flip the floor | at-floor green | UNPROVEN - nobody wrote it |"
    UNC_ROW="| Real gate | ${BT}scripts/real_guard.sh${BT} | flip the floor | at-floor green | UNCOVERED - not in the table |"

    row "R4 UNPROVEN beside a wired self-test" R4 "$(mk_root r4a "$UNP_ROW")"

    # A line continuation inside $( ) defeats bashrs's parser (SC1078 on a
    # string that is closed), so the workflow bodies are named first.
    local WF_PLAIN WF_COMMENTED
    WF_PLAIN='      - run: bash scripts/real_guard.sh\n'
    WF_COMMENTED='      - run: echo skipped # bash scripts/real_guard.sh --selftest\n'

    row "R4 UNCOVERED with no self-test anywhere" R4 "$(mk_root r4b "$UNC_ROW" "$WF_PLAIN")"

    row "R4 UNCOVERED beside a wired self-test" CLEAN \
        "$(mk_root r4d "| Real gate | ${BT}scripts/real_guard.sh${BT} | flip the floor | at-floor green | UNCOVERED - the table stays 9/9 |")"

    row "R4 a COMMENTED self-test is not coverage" CLEAN "$(mk_root r4c "$UNP_ROW" "$WF_COMMENTED")"

    row "R2 PROVEN naming a file not in the tree" R2 \
        "$(mk_root r2a "| Ghost gate | ${BT}scripts/no_such_guard.sh${BT} | flip the floor | at-floor green | PROVEN - 9/9 |")"

    row "R2 PROVEN naming no file at all" R2 \
        "$(mk_root r2b "| Ghost gate | some prose | flip the floor | at-floor green | PROVEN - 9/9 |")"

    row "R2 UNPROVEN may name no file" CLEAN \
        "$(mk_root r2c "| Ghost gate | verdict job | receipt one commit stale | fresh green | UNPROVEN - no such job |")"

    row "R1 a prose status is refused" R1 \
        "$(mk_root r1 "| Real gate | ${BT}scripts/real_guard.sh${BT} | flip the floor | at-floor green | **not written** |")"

    row "R3 an empty Mutation cell is refused" R3 \
        "$(mk_root r3a "| Real gate | ${BT}scripts/real_guard.sh${BT} |  | at-floor green | PROVEN - 9/9 |")"

    row "R3 an empty Discrimination cell is refused" R3 \
        "$(mk_root r3b "| Real gate | ${BT}scripts/real_guard.sh${BT} | flip the floor |  | PROVEN - 9/9 |")"

    # R0: the registry itself is gone. A scan that found nothing must be LOUD,
    # never "no violations".
    local r0="$td/r0"
    mkdir -p "$r0/docs/specifications" "$r0/.github/workflows"
    row "R0 no §5 table anywhere is a failure" R0 "$r0"

    # The vacuity floor is applied by the caller below, so prove the count it
    # reads is real rather than assuming it.
    local n root_vac
    root_vac="$(mk_root vac "$GOOD")"
    scan "$root_vac" > "$td/vac.tsv" 2>/dev/null
    n=$(rows_of "$td/vac.tsv" ANY | LC_ALL=C grep -c . || true)
    if [ "$n" = 1 ]; then
        printf '  ok    %-46s %s\n' "a one-row registry counts as one row" "1"
        pass=$((pass + 1))
    else
        printf '  BROKE %-46s expected 1 got %s\n' "row counting" "$n"
        fail=$((fail + 1))
    fi

    rm -rf "${td:?refusing to rm an empty path}"
    printf '  %d passed, %d broken\n' "$pass" "$fail"
    [ "$fail" = 0 ]
}

# -------------------------------------------------------------------- main --
main() {
    case "${1:-}" in
        --self-test|--selftest) selftest; return $? ;;
    esac

    printf -- '--- APR-PERF-GATE-001 §5 mutation registry vs the tree ------------\n'

    [ -f "$SCANNER" ] || { printf 'FAIL  %s is missing\n' "$SCANNER"; return 2; }

    local tsv rc=0
    tsv="$(mktemp)" || { printf 'FAIL  mktemp failed\n'; return 2; }
    cur="$(mktemp)" || { printf 'FAIL  mktemp failed\n'; return 2; }
    # shellcheck disable=SC2064  # expanded now on purpose: the paths are fixed here
    trap "rm -f '$tsv' '$cur'" EXIT

    if ! scan "$REPO_ROOT" > "$tsv"; then
        printf 'FAIL  the scanner errored; §5 is UNMEASURED, and an unmeasured\n'
        printf '      registry is not a registry. That is a failure, not a skip.\n'
        return 1
    fi

    local rows proven partial uncovered unproven viol
    rows=$(rows_of "$tsv" ANY | LC_ALL=C grep -c . || true)
    proven=$(rows_of "$tsv" PROVEN | LC_ALL=C grep -c . || true)
    partial=$(rows_of "$tsv" PARTIAL | LC_ALL=C grep -c . || true)
    uncovered=$(rows_of "$tsv" UNCOVERED | LC_ALL=C grep -c . || true)
    unproven=$(rows_of "$tsv" UNPROVEN | LC_ALL=C grep -c . || true)
    viol=$(LC_ALL=C awk -F'\t' '$1 == "VIOLATION"' "$tsv" | LC_ALL=C grep -c . || true)

    printf '%s row(s): %s PROVEN, %s PARTIAL, %s UNCOVERED, %s UNPROVEN\n' \
        "$rows" "$proven" "$partial" "$uncovered" "$unproven"

    if [ "$rows" -lt "$MIN_ROWS" ]; then
        printf '\nFAIL (vacuity): only %s row(s) parsed, expected %s+.\n' "$rows" "$MIN_ROWS"
        printf 'The scan is broken, or §5 was emptied. Fix that, not this number.\n'
        rc=1
    fi

    if [ "$viol" -gt 0 ]; then
        printf '\n%s row(s) disagree with the tree:\n' "$viol"
        render_violations "$tsv"
        rc=1
    fi

    not_proven_rows "$tsv" > "$cur"

    # THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE, and here the
    # REGISTRY IS ITS OWN BASELINE.
    #
    # The obvious build was a scripts/*_baseline.txt holding the not-PROVEN row
    # names, ratcheted with lib_baseline_ratchet.sh. It was built, and it is
    # wrong twice over. A brand-new baseline has no version at the comparand, so
    # the library's ABSENT branch hard-fails it — a new ratchet is born RED and
    # the only ways out are to disarm the branch or to land the file unratcheted
    # first, both of which are the disarm this epic removes. And a separate file
    # re-opens the laundering shape PERF-028 closed one level up: an entry plus
    # its matching downgrade in ONE commit.
    #
    # §5 is already a committed file on a ref no pull request can rewrite, so it
    # is the comparand. The comparand's own §5 is parsed with the SAME scanner,
    # its not-PROVEN set is taken, and the current set must be a SUBSET of it.
    # Removal is the point of a ratchet and stays green; a row that is newly
    # not-PROVEN — downgraded, or a new gate landing unproven — is refused, and
    # there is no in-branch file to record it in.
    #
    # Comparand resolution is lib_baseline_ratchet.sh's, not a third policy:
    # merge-base preferred, origin/main tip as the CI fallback, UNRESOLVABLE and
    # ABSENT both hard failures that never degrade to comparing the branch
    # against itself.
    # shellcheck source=scripts/lib_baseline_ratchet.sh
    . "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || return 1

    local spec_rel resolution mode ref base_root base_cur delta
    spec_rel=$(LC_ALL=C awk -F'\t' '$1 == "SPEC" { print $2; exit }' "$tsv")
    if [ -z "$spec_rel" ]; then
        printf '\nFAIL  the scanner named no spec file, so there is nothing to compare.\n'
        return 1
    fi

    resolution=$(baseline_ratchet_resolve "$REPO_ROOT" "$BASELINE_RATCHET_BASE_REF" "$spec_rel")
    mode=${resolution%%$'\t'*}
    ref=${resolution##*$'\t'}
    case "$mode" in
        UNRESOLVABLE)
            printf '\nFAIL  ratchet  cannot resolve <%s>, so §5 growth is UNMEASURED. It is\n' "$ref"
            printf '               NOT degraded to comparing this branch against itself.\n'
            printf '               In CI, before this guard runs:\n'
            printf '               git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main\n'
            return 1 ;;
        ABSENT)
            printf '\nFAIL  ratchet  %s carries no %s, so there is nothing to\n' "$ref" "$spec_rel"
            printf '               shrink from. A missing comparand is not "no growth". If the\n'
            printf '               spec was renamed, land the rename on main first.\n'
            return 1 ;;
    esac

    base_root="$(mktemp -d)" || return 2
    case "$base_root" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'FAIL  mktemp -d gave %s, refusing to rm -rf it\n' "${base_root:-<empty>}"; return 2 ;;
    esac
    mkdir -p "$base_root/$(dirname "$spec_rel")"
    if ! git -C "$REPO_ROOT" show "${ref}:${spec_rel}" > "$base_root/$spec_rel" 2>/dev/null; then
        rm -rf "${base_root:?}"
        printf '\nFAIL  ratchet  could not read %s:%s\n' "$ref" "$spec_rel"
        return 1
    fi
    base_cur="$(mktemp)" || { rm -rf "${base_root:?}"; return 2; }
    # The comparand is scanned with NO .github/workflows present, so R4 cannot
    # fire against it; only its ROW lines are read, and its own violations are
    # the previous commit's business, not this one's.
    scan "$base_root" > "$base_root/base.tsv" 2>/dev/null
    not_proven_rows "$base_root/base.tsv" > "$base_cur"

    delta=$(LC_ALL=C comm -13 "$base_cur" "$cur")
    local left
    left=$(LC_ALL=C comm -23 "$base_cur" "$cur" | LC_ALL=C grep -c . || true)
    rm -rf "${base_root:?}"

    if [ -n "$delta" ]; then
        printf '\nFAIL  ratchet  the not-PROVEN set GREW against %s:\n' \
            "$(git -C "$REPO_ROOT" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")"
        printf '%s\n' "$delta" | sed 's/^/        + /'
        printf '               A row may only LEAVE the not-PROVEN set. Prove it — apply the\n'
        printf '               mutation, quote the RED, the discrimination and the revert —\n'
        printf '               rather than recording that it is unproven.\n'
        rc=1
    else
        printf 'ok    ratchet  not-PROVEN set did not grow (%s left) vs %s\n' \
            "$left" \
            "$(git -C "$REPO_ROOT" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")"
        case "$mode" in
            MERGEBASE) printf '               comparand: %s §5 at merge-base with %s (protected)\n' \
                "$spec_rel" "$BASELINE_RATCHET_BASE_REF" ;;
            TIP)       printf '               comparand: %s §5 at the tip of %s (no merge-base; stricter)\n' \
                "$spec_rel" "$BASELINE_RATCHET_BASE_REF" ;;
        esac
        if [ "$BASELINE_RATCHET_BASE_REF" != "origin/main" ]; then
            printf '               OVERRIDDEN via BASELINE_RATCHET_BASE_REF — NOT a protected ref\n'
        fi
    fi

    printf '\n'
    if [ "$rc" -eq 0 ]; then
        printf 'PASS  every §5 row agrees with the tree, and the unproven set did not grow.\n'
    else
        printf 'FAIL  see rows above (#2752). §5 decides which arms §4.8 may read.\n'
    fi
    return "$rc"
}

main "$@"
