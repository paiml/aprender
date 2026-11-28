//! D5 and D6 of the §12 expiry DAG, and the DAG records, with the LAB exemption (PMAT-7534;
//! the port of `_dag_document`, `_compare_derived` and `check_dag` in
//! `scripts/lib/spec_conformance.py`).
//!
//! The one intended difference from the Python: a row whose expiry cell starts with the word
//! `LAB` is not a D6 andon when it is past its expiry. It emits one informational `LAB` record
//! instead (C324: a red that cannot block is LAB, and LAB cannot block a merge).

use crate::pystr::{py_path_str, py_read_text};
use crate::spec_conformance_dag::{section_12_lines, Violation};

/// One line of the scan's output: a DAG line, a VIOLATION, or the LAB record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    /// `DAG\t<row>\t<expires or ->\t<reason>`.
    Dag {
        key: String,
        expires: Option<String>,
        reason: String,
    },
    Violation(Violation),
    /// `LAB\t<row>\t<message>`: an expired LAB row, informational, never a VIOLATION.
    Lab {
        row: String,
        message: String,
    },
}

impl Record {
    /// The TSV line `emit()` writes for this record.
    #[must_use]
    pub fn line(&self) -> String {
        let _ = self;
        String::new()
    }

    #[must_use]
    pub fn is_violation(&self) -> bool {
        matches!(self, Self::Violation(_))
    }
}

/// The committed derived-expiry file: its path as the report names it, and its text (None when
/// the file is absent). The text is read the way Python's text mode reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Derived<'a> {
    pub rel: &'a str,
    pub committed: Option<&'a str>,
}

/// `expiry-cell` begins with the whole word `LAB` (after `**`/`*` were stripped).
#[must_use]
pub fn is_lab(expires_cell: &str) -> bool {
    let _ = expires_cell;
    false
}

/// `_dag_document` + `_compare_derived`: every record the §12 scan emits, in Python's order.
/// `spec` is the whole spec text, `source` the spec path as the report names it, `today` the
/// date D6 compares against (ISO `YYYY-MM-DD`).
#[must_use]
pub fn check_dag(
    spec: &str,
    source: &str,
    derived: Option<Derived<'_>>,
    today: &str,
) -> Vec<Record> {
    let _ = (spec, source, derived, today);
    Vec::new()
}

/// The derived-expiry file text (`json.dumps(..., indent=2, sort_keys=True) + "\n"`) for `spec`.
///
/// # Errors
///
/// The D0 violation when the §12 table cannot be parsed.
pub fn derived_text(spec: &str, source: &str) -> Result<String, Violation> {
    let _ = (spec, source);
    Ok(String::new())
}

/// `_compare_derived`: D5 when the committed file is absent or differs from `text`.
#[must_use]
pub fn compare_derived(derived: &Derived<'_>, text: &str) -> Vec<Record> {
    let _ = (derived, text);
    Vec::new()
}

/// The CLI: the records as TSV lines and the exit code (1 when any VIOLATION, else 0).
/// `today` None reads `SPEC_CONFORMANCE_TODAY`, else the system date.
///
/// # Errors
///
/// The reason when an input cannot be read (exit 2 at the caller).
pub fn run_cli(
    spec_path: &std::path::Path,
    derived_path: Option<&std::path::Path>,
    today: Option<String>,
) -> Result<(String, u8), String> {
    let _ = (spec_path, derived_path, today, py_read_text, py_path_str);
    Ok((String::new(), 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAIN: &str = include_str!("../tests/fixtures/spec_dag/pp_main_efa749e.md");
    const LAB: &str = include_str!("../tests/fixtures/spec_dag/pp_lab_9a10119.md");
    const DERIVED: &str = include_str!("../tests/fixtures/spec_dag/derived_main_efa749e.json");
    /// The path the CI gate reports the master under, and the one the derived file names.
    const SOURCE: &str = "docs/specifications/PP-LLAMA-001-MASTER.md";
    const REL: &str = "evidence/parity/derived_expiries.json";
    const TODAY: &str = "2026-11-28";
    /// The §12 rows that are past their expiry on `TODAY` on the main fixture.
    const EXPIRED: [&str; 8] = ["13", "15", "16", "17", "18", "19", "20", "21"];

    fn spec(rows: &str) -> String {
        format!(
            "## §12 Obligations\n\n| row | deliverable | blocked_by | status / expires |\n\
             |---|---|---|---|\n{rows}\n\n## Next\n"
        )
    }

    fn one_row(cell: &str) -> String {
        spec(&format!("| r1 | x | — | {cell} |"))
    }

    fn check(text: &str, today: &str) -> Vec<Record> {
        check_dag(text, SOURCE, None, today)
    }

    fn violations(recs: &[Record]) -> Vec<(&'static str, String)> {
        recs.iter()
            .filter_map(|r| match r {
                Record::Violation(v) => Some((v.rule, v.row.clone())),
                _ => None,
            })
            .collect()
    }

    fn lab_rows(recs: &[Record]) -> Vec<String> {
        recs.iter()
            .filter_map(|r| match r {
                Record::Lab { row, .. } => Some(row.clone()),
                _ => None,
            })
            .collect()
    }

    fn d6(row: &str) -> Vec<(&'static str, String)> {
        vec![("D6", row.to_owned())]
    }

    #[test]
    fn d6_fires_on_an_expired_open_row() {
        let recs = check(&one_row("**OPEN**. Expires **2026-11-01**"), TODAY);
        assert_eq!(violations(&recs), d6("r1"));
        assert!(lab_rows(&recs).is_empty());
    }

    #[test]
    fn d6_fires_when_lab_is_not_the_first_word() {
        let recs = check(
            &one_row("**OPEN**, see LAB notes. Expires **2026-11-01**"),
            TODAY,
        );
        assert_eq!(violations(&recs), d6("r1"));
    }

    #[test]
    fn d6_fires_on_label_and_collab_which_are_not_the_whole_word_lab() {
        for cell in [
            "**LABEL** (x). Expires **2026-11-01**",
            "**COLLAB** (x). Expires **2026-11-01**",
        ] {
            let recs = check(&one_row(cell), TODAY);
            assert_eq!(violations(&recs), d6("r1"), "cell: {cell}");
            assert!(lab_rows(&recs).is_empty(), "cell: {cell}");
        }
    }

    #[test]
    fn an_expired_lab_row_is_exempt_and_emits_one_lab_record() {
        let cell =
            "**LAB** (operator ruling, see Appendix D; was **OPEN**). Expires **2026-11-27**";
        let recs = check(&one_row(cell), TODAY);
        assert!(violations(&recs).is_empty(), "{recs:?}");
        assert_eq!(lab_rows(&recs), vec!["r1".to_owned()]);
        let lab = recs
            .iter()
            .find(|r| matches!(r, Record::Lab { .. }))
            .expect("one LAB record");
        assert_eq!(
            lab.line(),
            "LAB\tr1\texpired 2026-11-27 (today is 2026-11-28); LAB, cannot block (C324)"
        );
    }

    #[test]
    fn an_unexpired_open_row_fires_nothing() {
        let recs = check(&one_row("**OPEN**. Expires **2026-12-01**"), TODAY);
        assert!(violations(&recs).is_empty());
        assert!(lab_rows(&recs).is_empty());
    }

    #[test]
    fn a_landed_row_past_its_expiry_is_discharged_and_fires_nothing() {
        let recs = check(
            &one_row("**LANDED** (this PR). Expires **2026-01-01**"),
            TODAY,
        );
        assert!(violations(&recs).is_empty());
        assert!(lab_rows(&recs).is_empty());
    }

    #[test]
    fn d5_absent_derived_file_is_one_violation() {
        let d = Derived {
            rel: REL,
            committed: None,
        };
        let recs = compare_derived(&d, "{}\n");
        assert_eq!(violations(&recs), vec![("D5", "-".to_owned())]);
        assert!(recs[0].line().contains("is absent"), "{:?}", recs[0]);
    }

    #[test]
    fn d5_differing_derived_file_names_the_first_differing_line() {
        let d = Derived {
            rel: REL,
            committed: Some("a\nb\nc\n"),
        };
        let recs = compare_derived(&d, "a\nX\nc\n");
        assert_eq!(violations(&recs), vec![("D5", "-".to_owned())]);
        assert!(
            recs[0].line().contains("first difference at line 2)"),
            "{:?}",
            recs[0]
        );
    }

    #[test]
    fn d5_equal_derived_file_is_silent() {
        let d = Derived {
            rel: REL,
            committed: Some("a\nb\n"),
        };
        assert!(compare_derived(&d, "a\nb\n").is_empty());
    }

    #[test]
    fn section_12_lines_stop_at_the_next_heading() {
        let lines = section_12_lines("# Top\n## §12 X\n| a |\n## §13 Y\n| b |\n");
        assert_eq!(lines, vec!["| a |".to_owned()]);
    }

    #[test]
    fn derived_text_of_the_main_fixture_is_byte_identical_to_the_python_output() {
        let text = derived_text(MAIN, SOURCE).expect("§12 parses");
        assert_eq!(text, DERIVED);
    }

    #[test]
    fn parity_main_fixture_fires_d6_on_exactly_rows_13_and_15_to_21() {
        let d = Derived {
            rel: REL,
            committed: Some(DERIVED),
        };
        let recs = check_dag(MAIN, SOURCE, Some(d), TODAY);
        let want: Vec<(&'static str, String)> =
            EXPIRED.iter().map(|r| ("D6", (*r).to_owned())).collect();
        assert_eq!(violations(&recs), want);
        assert!(lab_rows(&recs).is_empty());
    }

    #[test]
    fn parity_lab_fixture_is_silent_on_d6_and_lab_records_the_same_rows() {
        let d = Derived {
            rel: REL,
            committed: Some(DERIVED),
        };
        let recs = check_dag(LAB, SOURCE, Some(d), TODAY);
        assert!(violations(&recs).is_empty(), "{recs:?}");
        let want: Vec<String> = EXPIRED.iter().map(|r| (*r).to_owned()).collect();
        assert_eq!(lab_rows(&recs), want);
    }
}
