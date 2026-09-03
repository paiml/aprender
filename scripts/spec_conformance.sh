#!/usr/bin/env bash
# spec_conformance.sh — PP-LLAMA-001-MASTER.md must describe the tree.
#
# WHY THIS EXISTS (PP-29)
# ----------------------
# It replaces scripts/check_mutation_registry.sh, which located its input by
# glob over `docs/specifications/APR-PERF-GATE-001-v*.md`. Archiving that
# document did not break the guard loudly: it made it read ZERO rows, report
# zero violations, and exit 1 with a vacuity message that looked like a path
# problem. A registry guard whose universe can empty itself is the shape the
# whole epic is about, so the successor is joined to a fixed path and refuses
# anything but exactly one match.
#
# THREE JOINS, one per way a spec can drift from the tree:
#
#   §6           every row whose status starts with ARMED names selftest cases
#                that EXIST, by name, on the surface the row declares. A
#                backticked name is not evidence; the name has to appear in a
#                case table something runs. Surfaces: `pg:` (perf_gate.sh's
#                --list-selftests), `sh:<script>:` (that script's own case
#                table), `rs:<crate>:` (a `#[test] fn` under crates/<crate>/src).
#   Appendix C   PP-9: a cell, once run, is SPENT. No two ledger rows may share
#                (host, workload, model quant, commit, interleaved) with
#                conformance RECORDED. Re-running a cell until it comes out
#                green is the defect, and it is only refusable against a record.
#   §12          expiries are DERIVED from the blocked_by DAG, never typed. A
#                root row carries a date; every other row's expiry is the LATEST
#                among its transitive blockers. Cycles are refused, and so is a
#                blocked row that types a date -- that is how an expiry comes to
#                fall before the work it waits on. The derivation is written to
#                evidence/parity/derived_expiries.json, which is what
#                perf-matrix.yaml's `expires_after:` anchors resolve against.
#
# WHAT IT DELIBERATELY DOES NOT CLAIM. It cannot read an English rule cell and
# decide the sentence describes something real. It joins NAMES to CASE TABLES;
# whether a case discriminates is the business of the table it lives in, and
# every one of those tables carries its own must-fire.
#
#   bash scripts/spec_conformance.sh              # check
#   bash scripts/spec_conformance.sh --selftest   # case table
#   bash scripts/spec_conformance.sh --list-selftests
#
# SPEC_CONFORMANCE_SPEC overrides the spec path (used while the master is being
# written, and by the case table below).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCANNER="${REPO_ROOT}/scripts/lib/spec_conformance.py"

# Vacuity floor. §6 holds 33 rules; a parser that silently stopped matching
# would report zero rows, zero violations and look like health -- the exact
# failure this guard is about.
MIN_ROWS="${MIN_ROWS:-33}"

scan() { # scan <root> [extra args...] -> TSV on stdout, non-zero if the scan broke
    local root="$1"
    shift
    python3 "$SCANNER" "$root" "$@"
}

render_violations() {
    LC_ALL=C awk -F'\t' '$1 == "VIOLATION" {
        printf "  %-4s %-28.28s %s\n", $2, $3, $4
    }' "$1"
}

render_notes() {
    LC_ALL=C awk -F'\t' '$1 == "NOTE" { printf "  note %s\n", $2 }' "$1"
}

# ---------------------------------------------------------------- selftest --
CASE_NAMES=()
LIST_ONLY="${SPEC_CONFORMANCE_LIST_ONLY:-0}"

_reg() {
    CASE_NAMES+=("$1")
    if [ "$LIST_ONLY" = 1 ]; then return 1; fi
    return 0
}

selftest() {
    # Declared HERE, before the nested helpers: `local` after a nested function
    # definition parses as top-level to bashrs (SC2168), and a lint error in a
    # guard is the one place a lint error is not cosmetic.
    local td pass=0 fail=0
    local BT ROW_ARMED ROW_OPEN DAG_OK LEDGER_ONE LEDGER_TWO LEDGER_TWO_COMMITS LEDGER_SPLIT
    local LEDGER_NOLEAD LEDGER_BACKTICK LEDGER_TRAILING LEDGER_EXTRA LEDGER_SIDE LEDGER_SUPERSEDED LEDGER_RESPEND_OUT
    local LEDGER_DUMMY_FIRST LEDGER_BOLD LEDGER_SIDE_SPLIT LEDGER_SIDE_WIDE LEDGER_SAME_HEADER LEDGER_EXTRA_OUT LEDGER_PRE
    local LEDGER_FOREIGN_HEADER LEDGER_THREE_OFF LEDGER_CONFORMANT_OUT LEDGER_NO_TIER_OUT
    local LEDGER_NONNUMERIC_ID LEDGER_BACKTICK_TIER LEDGER_RESERVED_WORD
    local LEDGER_UNDERSCORED_TIER LEDGER_RESPEND_CONFORMANT LEDGER_RESPEND_EMPHASISED_KEY LEDGER_RESPEND_ZWSP_KEY
    local root_dag derived
    td="$(mktemp -d)" || { printf 'FAIL  mktemp -d failed\n'; return 2; }
    case "$td" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'FAIL  mktemp -d gave %s, refusing to rm -rf it\n' "${td:-<empty>}"
           return 2 ;;
    esac

    # A synthetic root: a §6 table, a §12 table, a ledger and a stub whose
    # --list-selftests IS the surface. Building the surface rather than pointing
    # at the real one is what lets a row assert that a MISSING case reddens.
    mk_root() { # mk_root <name> <§6 rows> <§12 rows> <ledger rows> <stub case names> [<ledger preamble>]
        local r="$td/$1"
        rm -rf "${r:?}"
        mkdir -p "$r/docs/specifications" "$r/scripts" "$r/evidence/parity"
        {
            printf '#!/usr/bin/env bash\n'
            printf 'case "${1:-}" in\n'
            printf '  --list-selftests)\n'
            local n
            for n in $5; do printf '    printf %s\\\\n\n' "$n"; done
            printf '    ;;\n'
            printf '  --selftest) : ;;\n'
            printf 'esac\n'
        } > "$r/scripts/perf_gate.sh"
        {
            printf '# PP-LLAMA-001 (fixture)\n\n## §6 Invariants\n\n'
            printf '| id | rule | must-fire | must-not-fire | status | producer · selftest |\n'
            printf '|---|---|---|---|---|---|\n'
            printf '%b' "$2"
            printf '\n## §12 Owed work\n\n'
            printf '| row | what is owed | owner | blocked_by | expires |\n'
            printf '|---|---|---|---|---|\n'
            printf '%b' "$3"
            printf '\n## §13 End\n'
        } > "$r/docs/specifications/PP-LLAMA-001-MASTER.md"
        {
            printf '# Ledger (fixture)\n\n'
            printf '%b' "${6:-}"
            printf '| # | started_utc | host | class | model · quant | workload | commit | interleaved | receipts | conformance |\n'
            printf '|---|---|---|---|---|---|---|---|---|---|\n'
            printf '%b' "$4"
        } > "$r/evidence/parity/LEDGER.md"
        printf '%s' "$r"
    }

    row() { # row <name> <expected rules, comma-separated, or CLEAN> <root>
        _reg "$1" || return 0
        local name="$1" want="$2" root="$3" got
        if ! MIN_ROWS=1 scan "$root" --no-out > "$td/out.tsv" 2>"$td/err.txt"; then
            printf '  BROKE %-38s scanner errored: %s\n' "$name" "$(head -1 "$td/err.txt")"
            fail=$((fail + 1)); return 0
        fi
        got=$(LC_ALL=C awk -F'\t' '$1 == "VIOLATION" { print $2 }' "$td/out.tsv" \
              | LC_ALL=C sort -u | tr '\n' ',' | sed 's/,$//')
        [ -n "$got" ] || got=CLEAN
        if [ "$got" = "$want" ]; then
            printf '  ok    %-38s %s\n' "$name" "$want"; pass=$((pass + 1))
        else
            printf '  BROKE %-38s expected %s got %s\n' "$name" "$want" "$got"
            fail=$((fail + 1))
        fi
    }

    row_out() { # row_out <name> <expected rules> <root> -- like row, but COMPARES the committed derivation (no --no-out)
        _reg "$1" || return 0
        local name="$1" want="$2" root="$3" got
        if ! MIN_ROWS=1 scan "$root" --out "$root/evidence/parity/derived_expiries.json" > "$td/out.tsv" 2>"$td/err.txt"; then
            printf '  BROKE %-38s scanner errored: %s\n' "$name" "$(head -1 "$td/err.txt")"
            fail=$((fail + 1)); return 0
        fi
        got=$(LC_ALL=C awk -F'\t' '$1 == "VIOLATION" { print $2 }' "$td/out.tsv" \
              | LC_ALL=C sort -u | tr '\n' ',' | sed 's/,$//')
        [ -n "$got" ] || got=CLEAN
        if [ "$got" = "$want" ]; then
            printf '  ok    %-38s %s\n' "$name" "$want"; pass=$((pass + 1))
        else
            printf '  BROKE %-38s expected %s got %s\n' "$name" "$want" "$got"
            fail=$((fail + 1))
        fi
    }

    BT=$(printf '\140')          # a literal backtick, built rather than typed
    ROW_ARMED="| PP-1 | a rule | mutate | hold | ARMED | ${BT}scripts/x.sh${BT} · ${BT}pg:alpha${BT} / ${BT}pg:beta${BT} |\\n"
    ROW_OPEN="| PP-2 | a rule | mutate | hold | OPEN | ${BT}scripts/x.sh${BT} · ${BT}pg:gamma${BT} / ${BT}pg:delta${BT} |\\n"
    DAG_OK="| r1 | root a | o | — | 2026-01-01 |\\n| r2 | root b | o | — | 2026-03-01 |\\n| r3 | blocked | o | r1, r2 | derived |\\n"
    LEDGER_ONE="| 1 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | RECORDED |\\n"
    LEDGER_TWO="${LEDGER_ONE}| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | RECORDED |\\n"
    LEDGER_TWO_COMMITS="${LEDGER_ONE}| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_SPLIT="${LEDGER_ONE}\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_NOLEAD="${LEDGER_ONE} 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_BACKTICK="${LEDGER_ONE}\\n| ${BT}2${BT} | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_TRAILING="${LEDGER_ONE}| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e ||\\n"
    LEDGER_EXTRA="${LEDGER_ONE}| 2 | t | | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_SIDE="${LEDGER_TWO_COMMITS}\\n| c | agg |\\n|---|---|\\n| 1 | 90.2 |\\n"
    LEDGER_SUPERSEDED="${LEDGER_TWO_COMMITS}\\n## Superseded rows (schema v2)\\n\\n| 9 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | RECORDED |\\n"
    LEDGER_RESPEND_OUT="${LEDGER_ONE}\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | RECORDED |\\n"
    LEDGER_DUMMY_FIRST="${LEDGER_ONE}\\n| dummy | x |\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | RECORDED |\\n"
    LEDGER_BOLD="${LEDGER_ONE}\\n| **2** | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_SIDE_SPLIT="${LEDGER_TWO_COMMITS}\\n| c | agg |\\n|---|---|\\n| 1 | 90.2 |\\n\\n| 4 | 80.1 |\\n"
    LEDGER_SIDE_WIDE="${LEDGER_TWO_COMMITS}\\n| n | a | b | c | d | e | f | g | h | i |\\n|---|---|---|---|---|---|---|---|---|---|\\n| 1 | a | b | c | d | e | f | g | h | i |\\n"
    LEDGER_SAME_HEADER="${LEDGER_ONE}\\n| # | started_utc | host | class | model · quant | workload | commit | interleaved | receipts | conformance |\\n|---|---|---|---|---|---|---|---|---|---|\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_EXTRA_OUT="${LEDGER_ONE}\\n| 2 | t | | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_PRE="| # | repo |\\n|---|---|\\n| 1 | aprender |\\n\\n"
    LEDGER_FOREIGN_HEADER="${LEDGER_ONE}\\n| dummy | x |\\n|---|---|\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | RECORDED |\\n"
    LEDGER_THREE_OFF="${LEDGER_ONE}\\n| 2 | lambda | W1 | ${BT}bbbb${BT} | false | RECORDED |\\n"
    LEDGER_CONFORMANT_OUT="${LEDGER_ONE}\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | CONFORMANT |\\n"
    LEDGER_NO_TIER_OUT="${LEDGER_ONE}\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | probe, no run |\\n"
    LEDGER_NONNUMERIC_ID="${LEDGER_ONE}\\n| foo | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | RECORDED |\\n"
    LEDGER_BACKTICK_TIER="${LEDGER_ONE}\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}bbbb${BT} | false | e | ${BT}RECORDED${BT} |\\n"
    LEDGER_RESERVED_WORD="${LEDGER_TWO_COMMITS}\\n| # | repo | state |\\n|---|---|---|\\n| 3 | aprender | RECORDED as retained |\\n"
    LEDGER_UNDERSCORED_TIER="${LEDGER_ONE}| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | __RECORDED__ |\\n"
    LEDGER_RESPEND_CONFORMANT="| 1 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | CONFORMANT |\\n| 2 | t | lambda | cuda | q · q4 | W1 | ${BT}aaaa${BT} | false | e | CONFORMANT |\\n"
    LEDGER_RESPEND_EMPHASISED_KEY="${LEDGER_ONE}| 2 | t | __lambda__ | cuda | q · q4 | W1 | <code>aaaa</code> | false | e | RECORDED |\\n"
    LEDGER_RESPEND_ZWSP_KEY="${LEDGER_ONE}| 2 | t | lam\\u200bbda | cuda | q · q4 | W1 | ${BT}AAAA${BT} | false | e | RECORDED |\\n"

    # §6 -- the join itself.
    row conformance_ok CLEAN \
        "$(mk_root ok "$ROW_ARMED" "$DAG_OK" "$LEDGER_ONE" "alpha beta")"
    row conformance_missing C1 \
        "$(mk_root missing "$ROW_ARMED" "$DAG_OK" "$LEDGER_ONE" "alpha")"
    # AN EMPTY REGISTRY IS LOUD. A scan that found nothing must be a verdict row,
    # never "no violations".
    row conformance_vacuity C0 \
        "$(mk_root vacuity "" "$DAG_OK" "$LEDGER_ONE" "alpha beta")"
    # DISCRIMINATION -- an OPEN row names cases that do not exist, on purpose.
    # That is what OPEN MEANS, and a guard that reddened on it would push every
    # honest "not yet" into a false ARMED.
    row conformance_open_row_ignored CLEAN \
        "$(mk_root openrow "$ROW_ARMED$ROW_OPEN" "$DAG_OK" "$LEDGER_ONE" "alpha beta")"

    # §0.6 -- a retired rule keeps its number; a GAP is a deleted invariant.
    ROW_THREE="| PP-3 | a rule | mutate | hold | ARMED | ${BT}scripts/x.sh${BT} · ${BT}pg:alpha${BT} / ${BT}pg:beta${BT} |\\n"
    row conformance_gap_is_red C5 \
        "$(mk_root gap "$ROW_ARMED$ROW_THREE" "$DAG_OK" "$LEDGER_ONE" "alpha beta")"
    # The committed derivation must MATCH the table: a stale file is D5, and a
    # plain run never rewrites it (only --write does).
    root_drift="$(mk_root drift "$ROW_ARMED" "$DAG_OK" "$LEDGER_ONE" "alpha beta")"
    printf '{"rows": {}, "source": "stale"}\n' > "$root_drift/evidence/parity/derived_expiries.json"
    row_out dag_committed_drift_is_red D5 "$root_drift"
    root_fresh="$(mk_root fresh "$ROW_ARMED" "$DAG_OK" "$LEDGER_ONE" "alpha beta")"
    MIN_ROWS=1 scan "$root_fresh" --write > /dev/null 2>&1
    row_out dag_committed_matches_is_clean CLEAN "$root_fresh"

    # Appendix C -- PP-9, a cell once run is spent.
    row respend_same_key L1 \
        "$(mk_root respend "$ROW_ARMED" "$DAG_OK" "$LEDGER_TWO" "alpha beta")"
    row respend_new_commit CLEAN \
        "$(mk_root respend_ok "$ROW_ARMED" "$DAG_OK" "$LEDGER_TWO_COMMITS" "alpha beta")"
    # A blank line splits the ledger table: the second row shares the header's
    # column count and a row-id-shaped first cell, but table_rows() never sees
    # it -- L2, and only L2 (the two commits differ, so L1 does not also fire).
    row ledger_row_outside_table L2 \
        "$(mk_root split "$ROW_ARMED" "$DAG_OK" "$LEDGER_SPLIT" "alpha beta")"
    row ledger_rows_contiguous CLEAN \
        "$(mk_root contiguous "$ROW_ARMED" "$DAG_OK" "$LEDGER_TWO_COMMITS" "alpha beta")"
    # PMAT-931: the universe of L2 is the RULE, not a shape. A row without its
    # leading pipe ends the first table and starts a fragment (L2); a backticked
    # id is still an id (L2); a run opening with a header row is a different
    # table and is skipped whole (CLEAN); rows under the superseded heading are
    # out of scope (CLEAN); a fragment row re-spending a key fires L1 as well.
    row ledger_row_no_leading_pipe L2 \
        "$(mk_root nolead "$ROW_ARMED" "$DAG_OK" "$LEDGER_NOLEAD" "alpha beta")"
    row ledger_row_backticked_id L2 \
        "$(mk_root backtick "$ROW_ARMED" "$DAG_OK" "$LEDGER_BACKTICK" "alpha beta")"
    row ledger_side_table_ok CLEAN \
        "$(mk_root sidetable "$ROW_ARMED" "$DAG_OK" "$LEDGER_SIDE" "alpha beta")"
    row ledger_row_after_superseded_heading L1,L2 \
        "$(mk_root superseded "$ROW_ARMED" "$DAG_OK" "$LEDGER_SUPERSEDED" "alpha beta")"
    row ledger_respend_outside_table L1,L2 \
        "$(mk_root respendout "$ROW_ARMED" "$DAG_OK" "$LEDGER_RESPEND_OUT" "alpha beta")"
    # L3: a row inside the first table whose cell count differs from the header
    # shifts every column the spend key reads -- refused, not mis-keyed.
    row ledger_row_trailing_pipes L3 \
        "$(mk_root trailing "$ROW_ARMED" "$DAG_OK" "$LEDGER_TRAILING" "alpha beta")"
    row ledger_row_extra_pipe L3 \
        "$(mk_root extrapipe "$ROW_ARMED" "$DAG_OK" "$LEDGER_EXTRA" "alpha beta")"
    # PMAT-932: a run is read row by row unless it opens with a header that is
    # not the ledger's, so a dummy first line hides nothing; a bold id is an id;
    # a blank line inside a narrower side table flags nothing; a side table of
    # the ledger's own width is told apart by its header; a second table with
    # the ledger's header is outside the first table; a stray pipe outside the
    # table is still a ledger row; and a table written ABOVE the ledger is L0.
    row ledger_fragment_after_dummy_line L1,L2 \
        "$(mk_root dummyfirst "$ROW_ARMED" "$DAG_OK" "$LEDGER_DUMMY_FIRST" "alpha beta")"
    row ledger_row_bold_id L2 \
        "$(mk_root boldid "$ROW_ARMED" "$DAG_OK" "$LEDGER_BOLD" "alpha beta")"
    row ledger_side_table_split_ok CLEAN \
        "$(mk_root sidesplit "$ROW_ARMED" "$DAG_OK" "$LEDGER_SIDE_SPLIT" "alpha beta")"
    row ledger_side_table_same_width_ok CLEAN \
        "$(mk_root sidewide "$ROW_ARMED" "$DAG_OK" "$LEDGER_SIDE_WIDE" "alpha beta")"
    row ledger_second_table_same_header L2 \
        "$(mk_root sameheader "$ROW_ARMED" "$DAG_OK" "$LEDGER_SAME_HEADER" "alpha beta")"
    row ledger_row_extra_pipe_outside L2 \
        "$(mk_root extraout "$ROW_ARMED" "$DAG_OK" "$LEDGER_EXTRA_OUT" "alpha beta")"
    row ledger_first_table_not_ledger L0,L2 \
        "$(mk_root prefirst "$ROW_ARMED" "$DAG_OK" "$LEDGER_ONE" "alpha beta" "$LEDGER_PRE")"
    # PMAT-933: the universe is every row that CLAIMS A SPEND. A run under a
    # foreign header hides nothing; a row three columns off still claims its
    # tier; CONFORMANT is a spend too; a row that claims no tier spends nothing.
    row ledger_row_under_foreign_header L1,L2 \
        "$(mk_root foreignhdr "$ROW_ARMED" "$DAG_OK" "$LEDGER_FOREIGN_HEADER" "alpha beta")"
    row ledger_row_three_columns_off_outside L2 \
        "$(mk_root threeoff "$ROW_ARMED" "$DAG_OK" "$LEDGER_THREE_OFF" "alpha beta")"
    row ledger_row_conformant_outside L2 \
        "$(mk_root conformantout "$ROW_ARMED" "$DAG_OK" "$LEDGER_CONFORMANT_OUT" "alpha beta")"
    row ledger_row_without_tier_outside_ok CLEAN \
        "$(mk_root notier "$ROW_ARMED" "$DAG_OK" "$LEDGER_NO_TIER_OUT" "alpha beta")"
    # PMAT-934: the id is reported, never required; the tier is read through
    # the same normalisation as every other cell; RECORDED and CONFORMANT are
    # reserved words wherever they start a cell in this file, and no heading
    # ends the universe.
    row ledger_row_nonnumeric_id_outside L2 \
        "$(mk_root nonnumeric "$ROW_ARMED" "$DAG_OK" "$LEDGER_NONNUMERIC_ID" "alpha beta")"
    row ledger_row_backticked_tier_outside L2 \
        "$(mk_root backticktier "$ROW_ARMED" "$DAG_OK" "$LEDGER_BACKTICK_TIER" "alpha beta")"
    row ledger_reserved_word_in_side_table L2 \
        "$(mk_root reservedword "$ROW_ARMED" "$DAG_OK" "$LEDGER_RESERVED_WORD" "alpha beta")"
    # PMAT-935: one normaliser for every cell. Underscore emphasis around the
    # tier, CONFORMANT as a spend for L1 too, emphasis, code tags, a zero-width
    # space or letter case inside a key -- none of them makes a second run new.
    row ledger_row_underscored_tier L1 \
        "$(mk_root underscoredtier "$ROW_ARMED" "$DAG_OK" "$LEDGER_UNDERSCORED_TIER" "alpha beta")"
    row ledger_respend_conformant L1 \
        "$(mk_root respendconformant "$ROW_ARMED" "$DAG_OK" "$LEDGER_RESPEND_CONFORMANT" "alpha beta")"
    row ledger_respend_emphasised_key L1 \
        "$(mk_root emphkey "$ROW_ARMED" "$DAG_OK" "$LEDGER_RESPEND_EMPHASISED_KEY" "alpha beta")"
    row ledger_respend_zero_width_key L1 \
        "$(mk_root zwspkey "$ROW_ARMED" "$DAG_OK" "$LEDGER_RESPEND_ZWSP_KEY" "alpha beta")"

    # §12 -- the expiry DAG.
    if _reg dag_derived_equals_max_blocker; then
        root_dag="$(mk_root dagmax "$ROW_ARMED" "$DAG_OK" "$LEDGER_ONE" "alpha beta")"
        MIN_ROWS=1 scan "$root_dag" --no-out > "$td/dag.tsv" 2>/dev/null
        derived=$(LC_ALL=C awk -F'\t' '$1 == "DAG" && $2 == "r3" { print $3 }' "$td/dag.tsv")
        if [ "$derived" = 2026-03-01 ]; then
            printf '  ok    %-38s %s\n' dag_derived_equals_max_blocker "r3 = max(r1, r2) = 2026-03-01"
            pass=$((pass + 1))
        else
            printf '  BROKE %-38s expected 2026-03-01 got [%s]\n' dag_derived_equals_max_blocker "$derived"
            fail=$((fail + 1))
        fi
    fi
    row dag_cycle_is_red D1 \
        "$(mk_root dagcycle "$ROW_ARMED" \
           "| r1 | a | o | r2 | derived |\\n| r2 | b | o | r1 | derived |\\n" \
           "$LEDGER_ONE" "alpha beta")"
    row dag_root_without_date_is_red D2 \
        "$(mk_root dagroot "$ROW_ARMED" \
           "| r1 | a | o | — | derived |\\n" "$LEDGER_ONE" "alpha beta")"
    row dag_nonroot_with_date_is_red D3 \
        "$(mk_root dagnonroot "$ROW_ARMED" \
           "| r1 | a | o | — | 2026-01-01 |\\n| r2 | b | o | r1 | 2025-01-01 |\\n" \
           "$LEDGER_ONE" "alpha beta")"

    if [ "$LIST_ONLY" = 1 ]; then
        rm -rf "${td:?refusing to rm an empty path}"
        return 0
    fi
    rm -rf "${td:?refusing to rm an empty path}"
    printf '  %d passed, %d broken\n' "$pass" "$fail"
    [ "$fail" = 0 ]
}

# -------------------------------------------------------------------- main --
main() {
    case "${1:-}" in
        # A LITERAL `--selftest)` ARM, on its own line, and not folded into
        # `--selftest|--self-test)`. check_guards_are_wired.sh derives its
        # universe from `^\s*(--selftest|--self-test)\)`, so the folded spelling
        # is invisible to it and this script -- which is not named check_* --
        # could go dark without the meta-guard noticing. That is the exact defect
        # the meta-guard exists to catch, one level down.
        --selftest) selftest; return $? ;;
        # Regenerate evidence/parity/derived_expiries.json from §12. The plain
        # run COMPARES against the committed file (D5) and never writes.
        --write) scan "$REPO_ROOT" --write > /dev/null; return $? ;;
        --self-test) selftest; return $? ;;
        --list-selftests)
            SPEC_CONFORMANCE_LIST_ONLY=1 LIST_ONLY=1 selftest >/dev/null 2>&1
            printf '%s\n' "${CASE_NAMES[@]}"
            return 0 ;;
    esac

    printf -- '--- PP-LLAMA-001 §6 / Appendix C / §12 vs the tree ----------------\n'

    [ -f "$SCANNER" ] || { printf 'FAIL  %s is missing\n' "$SCANNER"; return 2; }

    local tsv rc=0 args=()
    tsv="$(mktemp)" || { printf 'FAIL  mktemp failed\n'; return 2; }
    # shellcheck disable=SC2064  # expanded now on purpose: the path is fixed here
    trap "rm -f '$tsv'" EXIT
    if [ -n "${SPEC_CONFORMANCE_SPEC:-}" ]; then
        args+=(--spec "$SPEC_CONFORMANCE_SPEC")
    fi

    if ! scan "$REPO_ROOT" "${args[@]}" > "$tsv"; then
        printf 'FAIL  the scanner errored; §6 is UNMEASURED, and an unmeasured\n'
        printf '      registry is not a registry. That is a failure, not a skip.\n'
        return 1
    fi

    local spec rows armed cases missing viol parsed
    spec=$(LC_ALL=C awk -F'\t' '$1 == "SPEC" { print $2; exit }' "$tsv")
    parsed=$(LC_ALL=C awk -F'\t' '$1 == "PARSED" { print $2; exit }' "$tsv")
    rows=$(LC_ALL=C awk -F'\t' '$1 == "ROW"' "$tsv" | LC_ALL=C grep -c . || true)
    armed=$(LC_ALL=C awk -F'\t' '$1 == "ROW" && toupper($2) ~ /^ARMED/' "$tsv" \
            | LC_ALL=C grep -c . || true)
    cases=$(LC_ALL=C awk -F'\t' '$1 == "CASE"' "$tsv" | LC_ALL=C grep -c . || true)
    missing=$(LC_ALL=C awk -F'\t' '$1 == "CASE" && $4 == "missing"' "$tsv" \
              | LC_ALL=C grep -c . || true)
    viol=$(LC_ALL=C awk -F'\t' '$1 == "VIOLATION"' "$tsv" | LC_ALL=C grep -c . || true)

    printf 'spec: %s\n' "${spec:-<none>}"
    printf '%s row(s), %s ARMED, %s named case(s), %s missing\n' \
        "$rows" "$armed" "$cases" "$missing"
    render_notes "$tsv"

    if [ "${parsed:-0}" -lt "$MIN_ROWS" ]; then
        printf '\nFAIL (vacuity): only %s row(s) parsed, expected %s+.\n' "${parsed:-0}" "$MIN_ROWS"
        printf 'The scan is broken, or §6 was emptied. Fix that, not this number.\n'
        rc=1
    fi

    if [ "$viol" -gt 0 ]; then
        printf '\n%s row(s) disagree with the tree:\n' "$viol"
        render_violations "$tsv"
        rc=1
    fi

    printf '\n'
    if [ "$rc" -eq 0 ]; then
        printf 'PASS  every ARMED row names cases that exist, no cell was spent twice,\n'
        printf '      and every §12 expiry derives from its blockers.\n'
    else
        printf 'FAIL  see rows above. §6 decides which rules the verdict may read.\n'
    fi
    return "$rc"
}

main "$@"
