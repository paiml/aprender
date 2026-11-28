//! L0-L3 of the PP-9 ledger (PMAT-7534; the port of `check_ledger` and its helpers in
//! `scripts/lib/spec_conformance.py`, with the selftest's `LEDGER_*` fixtures as unit tests).
//!
//! PP-9: a cell, once run, is SPENT. A re-run that comes out green is the defect, and it is
//! refusable only against the written ledger. L0 is the ledger that cannot be read as a
//! ledger, L1 a re-spent key, L2 a spend row outside the table PP-9 reads, L3 a malformed row.

use crate::spec_conformance_dag::Violation;

/// One line of the ledger scan's output: a VIOLATION, the `LEDGER` summary, or a `NOTE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    Violation(Violation),
    /// `LEDGER\t<rows>\t<recorded>`: the universe size and how many rows are RECORDED.
    Ledger {
        rows: usize,
        recorded: usize,
    },
    /// `NOTE\t<text>`.
    Note(String),
}

impl Record {
    /// The TSV line `emit()` writes for this record.
    #[must_use]
    pub fn line(&self) -> String {
        match self {
            Self::Violation(v) => v.line(),
            Self::Ledger { rows, recorded } => format!("LEDGER\t{rows}\t{recorded}"),
            Self::Note(text) => format!("NOTE\t{}", text.replace('\t', " ")),
        }
    }
}

/// Every record the ledger rules emit for one ledger file, in Python's order. `path_label` is the
/// ledger path as the report names it; `text` is the file's contents, or None when it is absent.
#[must_use]
pub fn check_ledger(path_label: &str, text: Option<&str>) -> Vec<Record> {
    let _ = (path_label, text);
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The label the selftest's ledger lives under.
    const LABEL: &str = "evidence/parity/LEDGER.md";
    const HEADER: &str = "| # | started_utc | host | class | model · quant | workload | commit | interleaved | receipts | conformance |\n|---|---|---|---|---|---|---|---|---|---|\n";
    const MAIN: &str = include_str!("../tests/fixtures/spec_ledger/ledger_main_efa749e.md");

    /// `mk_root`'s ledger file: a title, the optional preamble, the header, then the rows.
    fn ledger(preamble: &str, rows: &str) -> String {
        format!("# Ledger (fixture)\n\n{preamble}{HEADER}{rows}")
    }

    fn row(id: &str, commit: &str, tier: &str) -> String {
        format!("| {id} | t | lambda | cuda | q · q4 | W1 | `{commit}` | false | e | {tier} |\n")
    }

    fn one() -> String {
        row("1", "aaaa", "RECORDED")
    }

    fn two_same() -> String {
        one() + &row("2", "aaaa", "RECORDED")
    }

    fn two_commits() -> String {
        one() + &row("2", "bbbb", "RECORDED")
    }

    /// The selftest's `row` helper: the sorted, unique rule names, or CLEAN.
    fn rules(text: &str) -> String {
        let mut rules: Vec<&str> = check_ledger(LABEL, Some(text))
            .iter()
            .filter_map(|r| match r {
                Record::Violation(v) => Some(v.rule),
                _ => None,
            })
            .collect();
        rules.sort_unstable();
        rules.dedup();
        if rules.is_empty() {
            "CLEAN".to_owned()
        } else {
            rules.join(",")
        }
    }

    // ---- L0: the ledger cannot be read as a ledger --------------------------------------

    #[test]
    fn ledger_absent_is_l0() {
        assert_eq!(
            check_ledger(LABEL, None),
            vec![Record::Violation(Violation {
                rule: "L0",
                row: "-".to_owned(),
                message: "evidence/parity/LEDGER.md is absent -- PP-9 ('a cell, once run, is \
                          spent') is only enforceable against a written record of what has \
                          been spent"
                    .to_owned(),
            })]
        );
    }

    #[test]
    fn ledger_without_a_table_is_l0_and_nothing_else() {
        let records = check_ledger(LABEL, Some("# Ledger\n\nprose only, no pipe table\n"));
        assert_eq!(
            records,
            vec![Record::Violation(Violation {
                rule: "L0",
                row: "-".to_owned(),
                message: "no row table parsed from the ledger".to_owned(),
            })]
        );
    }

    #[test]
    fn empty_ledger_file_is_l0() {
        assert_eq!(rules(""), "L0");
    }

    #[test]
    fn header_missing_columns_is_l0_listing_them_sorted() {
        let text = "# L\n\n| # | repo |\n|---|---|\n| 1 | aprender |\n";
        let messages: Vec<String> = check_ledger(LABEL, Some(text))
            .iter()
            .filter_map(|r| match r {
                Record::Violation(v) if v.rule == "L0" => Some(v.message.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0].contains(
                "carries no commit, conformance, host, interleaved, model, workload column"
            ),
            "{}",
            messages[0]
        );
    }

    #[test]
    fn first_table_not_ledger_is_l0_and_l2() {
        let pre = "| # | repo |\n|---|---|\n| 1 | aprender |\n\n";
        assert_eq!(rules(&ledger(pre, &one())), "L0,L2");
    }

    // ---- L1: a spend key that is spent twice ---------------------------------------------

    #[test]
    fn respend_same_key() {
        assert_eq!(rules(&ledger("", &two_same())), "L1");
    }

    #[test]
    fn respend_new_commit() {
        assert_eq!(rules(&ledger("", &two_commits())), "CLEAN");
    }

    #[test]
    fn ledger_respend_outside_table() {
        assert_eq!(
            rules(&ledger("", &(one() + "\n" + &row("2", "aaaa", "RECORDED")))),
            "L1,L2"
        );
    }

    #[test]
    fn ledger_row_after_superseded_heading() {
        let rows = two_commits()
            + "\n## Superseded rows (schema v2)\n\n| 9 | t | lambda | cuda | q · q4 | W1 | `aaaa` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L1,L2");
    }

    #[test]
    fn ledger_fragment_after_dummy_line() {
        let rows = one() + "\n| dummy | x |\n" + &row("2", "aaaa", "RECORDED");
        assert_eq!(rules(&ledger("", &rows)), "L1,L2");
    }

    #[test]
    fn ledger_row_under_foreign_header() {
        let rows = one() + "\n| dummy | x |\n|---|---|\n" + &row("2", "aaaa", "RECORDED");
        assert_eq!(rules(&ledger("", &rows)), "L1,L2");
    }

    #[test]
    fn ledger_respend_conformant() {
        let rows = row("1", "aaaa", "CONFORMANT") + &row("2", "aaaa", "CONFORMANT");
        assert_eq!(rules(&ledger("", &rows)), "L1");
    }

    #[test]
    fn ledger_row_underscored_tier() {
        let rows =
            one() + "| 2 | t | lambda | cuda | q · q4 | W1 | `aaaa` | false | e | __RECORDED__ |\n";
        assert_eq!(rules(&ledger("", &rows)), "L1");
    }

    #[test]
    fn ledger_respend_emphasised_key() {
        let rows = one() + "| 2 | t | __lambda__ | cuda | q · q4 | W1 | <code>aaaa</code> | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L1");
    }

    #[test]
    fn ledger_respend_zero_width_key() {
        let rows = one()
            + "| 2 | t | lam\u{200b}bda | cuda | q · q4 | W1 | `AAAA` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L1");
    }

    // ---- L2: a spend row outside the table PP-9 reads ------------------------------------

    #[test]
    fn ledger_row_outside_table() {
        let rows = one() + "\n" + &row("2", "bbbb", "RECORDED");
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_rows_contiguous() {
        assert_eq!(rules(&ledger("", &two_commits())), "CLEAN");
    }

    #[test]
    fn ledger_row_no_leading_pipe() {
        let rows =
            one() + " 2 | t | lambda | cuda | q · q4 | W1 | `bbbb` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_row_backticked_id() {
        let rows =
            one() + "\n| `2` | t | lambda | cuda | q · q4 | W1 | `bbbb` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_side_table_ok() {
        let rows = two_commits() + "\n| c | agg |\n|---|---|\n| 1 | 90.2 |\n";
        assert_eq!(rules(&ledger("", &rows)), "CLEAN");
    }

    #[test]
    fn ledger_row_bold_id() {
        let rows = one()
            + "\n| **2** | t | lambda | cuda | q · q4 | W1 | `bbbb` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_side_table_split_ok() {
        let rows = two_commits() + "\n| c | agg |\n|---|---|\n| 1 | 90.2 |\n\n| 4 | 80.1 |\n";
        assert_eq!(rules(&ledger("", &rows)), "CLEAN");
    }

    #[test]
    fn ledger_side_table_same_width_ok() {
        let rows = two_commits()
            + "\n| n | a | b | c | d | e | f | g | h | i |\n|---|---|---|---|---|---|---|---|---|---|\n| 1 | a | b | c | d | e | f | g | h | i |\n";
        assert_eq!(rules(&ledger("", &rows)), "CLEAN");
    }

    #[test]
    fn ledger_second_table_same_header() {
        let rows = one() + "\n" + HEADER + &row("2", "bbbb", "RECORDED");
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_row_extra_pipe_outside() {
        let rows =
            one() + "\n| 2 | t | | lambda | cuda | q · q4 | W1 | `bbbb` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_row_three_columns_off_outside() {
        let rows = one() + "\n| 2 | lambda | W1 | `bbbb` | false | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_row_conformant_outside() {
        let rows = one() + "\n" + &row("2", "bbbb", "CONFORMANT");
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_row_without_tier_outside_ok() {
        let rows = one() + "\n" + &row("2", "bbbb", "probe, no run");
        assert_eq!(rules(&ledger("", &rows)), "CLEAN");
    }

    #[test]
    fn ledger_row_nonnumeric_id_outside() {
        let rows =
            one() + "\n| foo | t | lambda | cuda | q · q4 | W1 | `bbbb` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_row_backticked_tier_outside() {
        let rows =
            one() + "\n| 2 | t | lambda | cuda | q · q4 | W1 | `bbbb` | false | e | `RECORDED` |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    #[test]
    fn ledger_reserved_word_in_side_table() {
        let rows = two_commits()
            + "\n| # | repo | state |\n|---|---|---|\n| 3 | aprender | RECORDED as retained |\n";
        assert_eq!(rules(&ledger("", &rows)), "L2");
    }

    // ---- L3: a row inside the first table with the wrong cell count ----------------------

    #[test]
    fn ledger_row_trailing_pipes() {
        let rows = one() + "| 2 | t | lambda | cuda | q · q4 | W1 | `bbbb` | false | e ||\n";
        assert_eq!(rules(&ledger("", &rows)), "L3");
    }

    #[test]
    fn ledger_row_extra_pipe() {
        let rows =
            one() + "| 2 | t | | lambda | cuda | q · q4 | W1 | `bbbb` | false | e | RECORDED |\n";
        assert_eq!(rules(&ledger("", &rows)), "L3");
    }

    #[test]
    fn l3_message_names_the_row_and_both_widths() {
        let rows = one() + "| 2 | t | lambda | cuda | q · q4 | W1 | `bbbb` | false | e ||\n";
        let records = check_ledger(LABEL, Some(&ledger("", &rows)));
        assert!(records.contains(&Record::Violation(Violation {
            rule: "L3",
            row: "2".to_owned(),
            message: "ledger row 2 has 9 cell(s) against a 10-cell header: a malformed row shifts \
                      every column the spend key reads, so it is refused rather than mis-keyed"
                .to_owned(),
        })));
    }

    // ---- the LEDGER summary and NOTE -----------------------------------------------------

    #[test]
    fn ledger_summary_counts_the_universe_and_the_recorded_rows() {
        assert_eq!(
            check_ledger(LABEL, Some(&ledger("", &one()))),
            vec![Record::Ledger {
                rows: 1,
                recorded: 1
            }]
        );
    }

    #[test]
    fn ledger_without_a_recorded_row_adds_the_note() {
        let records = check_ledger(LABEL, Some(&ledger("", &row("1", "aaaa", "probe, no run"))));
        assert_eq!(
            records,
            vec![
                Record::Ledger {
                    rows: 1,
                    recorded: 0
                },
                Record::Note(
                    "no ledger row is marked conformance RECORDED, so the PP-9 duplicate rule \
                     matched nothing on this tree; its must-fire lives in the fixture rows of \
                     --selftest"
                        .to_owned()
                ),
            ]
        );
    }

    #[test]
    fn record_lines_are_the_tsv_the_guard_reads() {
        assert_eq!(
            Record::Ledger {
                rows: 12,
                recorded: 12
            }
            .line(),
            "LEDGER\t12\t12"
        );
        assert_eq!(Record::Note("a\tb".to_owned()).line(), "NOTE\ta b");
    }

    // ---- the parity measurement on main -------------------------------------------------

    /// `SPEC_CONFORMANCE_TODAY=2026-11-28 python3 scripts/lib/spec_conformance.py . --no-out`
    /// on the tree at d78e2e84b, which carries the same evidence/parity/LEDGER.md as origin/main
    /// efa749e19: the LEDGER line is `LEDGER\t12\t12` and no L-rule line is printed.
    #[test]
    fn main_ledger_matches_the_python_measurement() {
        let records = check_ledger(LABEL, Some(MAIN));
        let lines: Vec<String> = records.iter().map(Record::line).collect();
        assert_eq!(lines, vec!["LEDGER\t12\t12".to_owned()]);
    }
}
