//! D5 and D6 of the §12 expiry DAG, and the DAG records, with the LAB exemption (PMAT-7534;
//! the port of `_dag_document`, `_compare_derived` and `check_dag` in
//! `scripts/lib/spec_conformance.py`).
//!
//! The one intended difference from the Python: a row whose expiry cell starts with the word
//! `LAB` is not a D6 andon when it is past its expiry. It emits one informational `LAB` record
//! instead (C324: a red that cannot block is LAB, and LAB cannot block a merge).

use crate::pyjson::{dumps, Py};
use crate::pystr::{py_path_str, py_read_text, py_splitlines};
use crate::spec_conformance_dag::{
    analyze, from_markdown_table, section_12_lines, Analysis, Dag, Expiry, Row, Violation,
};
use std::path::Path;
use std::process::Command;

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
        match self {
            Self::Dag {
                key,
                expires,
                reason,
            } => tsv(&["DAG", key, expires.as_deref().unwrap_or("-"), reason]),
            Self::Violation(v) => v.line(),
            Self::Lab { row, message } => tsv(&["LAB", row, message]),
        }
    }

    #[must_use]
    pub fn is_violation(&self) -> bool {
        matches!(self, Self::Violation(_))
    }
}

/// `emit()`: the fields joined by tabs, with a tab inside a field replaced by a space.
fn tsv(fields: &[&str]) -> String {
    fields
        .iter()
        .map(|f| f.replace('\t', " "))
        .collect::<Vec<_>>()
        .join("\t")
}

/// The committed derived-expiry file: its path as the report names it, and its text (None when
/// the file is absent). The text is read the way Python's text mode reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Derived<'a> {
    pub rel: &'a str,
    pub committed: Option<&'a str>,
}

/// The expiry cell's first word, after `**`/`*` are stripped, is exactly `LAB`. A word is the
/// run of letters, digits, `_` and `-` at the start, so `LAB,` is LAB and `LABEL` or `LAB-2`
/// are not.
#[must_use]
pub fn is_lab(expires_cell: &str) -> bool {
    let plain = expires_cell.replace('*', "");
    let word: String = plain
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    word == "LAB"
}

/// A LAB cell is exempt from D6 only if it cites the ruling that moved it to LAB: `Appendix D`,
/// where the spec records operator rulings, and a ticket `#<digits>`, the one ticket C324 gives
/// every LAB row. For PMAT-7534 the ruling is the operator's of 2026-11-28 (Appendix D 3.1),
/// which covers rows 13 and 15-21. A bare `LAB`, or one that cites only one of the two, is still
/// an andon, so marking a row LAB cannot clear its D6 by itself.
#[must_use]
pub fn cites_lab_ruling(expires_cell: &str) -> bool {
    let names_ticket = expires_cell
        .match_indices('#')
        .any(|(i, _)| expires_cell[i + 1..].starts_with(|c: char| c.is_ascii_digit()));
    expires_cell.contains("Appendix D") && names_ticket
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
    let (dag, analysis) = match build(spec) {
        Ok(built) => built,
        Err(violation) => return vec![Record::Violation(violation)],
    };
    // The walk's D1-D4 come first, as Python emits them during derive_expiries.
    let mut out: Vec<Record> = analysis
        .violations
        .iter()
        .cloned()
        .map(Record::Violation)
        .collect();
    out.extend(document_records(&dag, &analysis, today));
    if let Some(d) = derived {
        out.extend(compare_derived(&d, &render(source, &dag, &analysis)));
    }
    out
}

fn build(spec: &str) -> Result<(Dag, Analysis), Violation> {
    let dag = from_markdown_table(&section_12_lines(spec).join("\n"))?;
    let analysis = analyze(&dag);
    Ok((dag, analysis))
}

/// The row indexes in sorted key order (`sorted(table)`).
fn sorted_indexes(rows: &[Row]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|a, b| rows[*a].key.cmp(&rows[*b].key));
    order
}

/// Each row's DAG line, followed by its D6 (or LAB) record if it is past its expiry.
fn document_records(dag: &Dag, analysis: &Analysis, today: &str) -> Vec<Record> {
    let rows = dag.rows();
    let mut out = Vec::new();
    for idx in sorted_indexes(rows) {
        let row = &rows[idx];
        let (_, exp) = &analysis.expiries[idx];
        out.push(Record::Dag {
            key: row.key.clone(),
            expires: exp.value.clone(),
            reason: dag_reason(row, exp),
        });
        out.extend(andon(row, exp, today));
    }
    out
}

/// `_dag_reason`: the blockers the expiry came from, else discharged, root or unblocked.
fn dag_reason(row: &Row, exp: &Expiry) -> String {
    if !exp.via.is_empty() {
        exp.via.join(",")
    } else if exp.discharged {
        "discharged".to_owned()
    } else if row.blocked_by.is_empty() {
        "root".to_owned()
    } else {
        "unblocked".to_owned()
    }
}

/// D6: a live row past its derived expiry. A LAB row past its expiry is a LAB record instead.
fn andon(row: &Row, exp: &Expiry, today: &str) -> Option<Record> {
    let date = exp.value.as_deref().filter(|d| !d.is_empty())?;
    if exp.discharged || date >= today {
        return None;
    }
    if is_lab(&row.expires_cell) && cites_lab_ruling(&row.expires_cell) {
        return Some(Record::Lab {
            row: row.key.clone(),
            message: format!("expired {date} (today is {today}); LAB, cannot block (C324)"),
        });
    }
    Some(Record::Violation(Violation {
        rule: "D6",
        row: row.key.clone(),
        message: format!(
            "expired {date} (today is {today}) and is not discharged. §4's andon \
             refuses a row past its expiry with nothing red for it"
        ),
    }))
}

/// The derived-expiry file text (`json.dumps(..., indent=2, sort_keys=True) + "\n"`) for `spec`.
///
/// # Errors
///
/// The D0 violation when the §12 table cannot be parsed.
pub fn derived_text(spec: &str, source: &str) -> Result<String, Violation> {
    let (dag, analysis) = build(spec)?;
    Ok(render(source, &dag, &analysis))
}

/// `{"rows": {...}, "source": ...}` with `sort_keys` and `indent=2`, as `_dag_document` writes it.
fn render(source: &str, dag: &Dag, analysis: &Analysis) -> String {
    let rows = dag.rows();
    let entries: Vec<(String, Py)> = sorted_indexes(rows)
        .into_iter()
        .map(|idx| {
            (
                rows[idx].key.clone(),
                dag_entry(&rows[idx], &analysis.expiries[idx].1),
            )
        })
        .collect();
    let document = Py::Dict(vec![
        ("rows".to_owned(), Py::Dict(entries)),
        ("source".to_owned(), Py::Str(source.to_owned())),
    ]);
    let mut out = String::new();
    write_indented(&document, 0, &mut out);
    out.push('\n');
    out
}

/// One row's entry; its keys are already in sorted order.
fn dag_entry(row: &Row, exp: &Expiry) -> Py {
    Py::Dict(vec![
        ("blocked_by".to_owned(), strings(&row.blocked_by)),
        ("derived_from".to_owned(), strings(&exp.via)),
        ("discharged".to_owned(), Py::Bool(exp.discharged)),
        (
            "expires".to_owned(),
            exp.value.clone().map_or(Py::None, Py::Str),
        ),
        ("root".to_owned(), Py::Bool(row.blocked_by.is_empty())),
    ])
}

fn strings(items: &[String]) -> Py {
    Py::List(items.iter().map(|s| Py::Str(s.clone())).collect())
}

/// `json.dumps(v, indent=2)` for the document: non-empty containers are laid out one item per
/// line, and every scalar is `dumps`' own text (its string escaping is the same).
fn write_indented(v: &Py, level: usize, out: &mut String) {
    match v {
        Py::Dict(kv) if !kv.is_empty() => {
            out.push('{');
            for (i, (key, item)) in kv.iter().enumerate() {
                open_item(i, level, out);
                out.push_str(&dumps(&Py::Str(key.clone())));
                out.push_str(": ");
                write_indented(item, level + 1, out);
            }
            close_container(level, '}', out);
        }
        Py::List(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                open_item(i, level, out);
                write_indented(item, level + 1, out);
            }
            close_container(level, ']', out);
        }
        _ => out.push_str(&dumps(v)),
    }
}

fn open_item(i: usize, level: usize, out: &mut String) {
    if i > 0 {
        out.push(',');
    }
    out.push('\n');
    out.push_str(&"  ".repeat(level + 1));
}

fn close_container(level: usize, close: char, out: &mut String) {
    out.push('\n');
    out.push_str(&"  ".repeat(level));
    out.push(close);
}

/// `_compare_derived`: D5 when the committed file is absent or differs from `text`.
#[must_use]
pub fn compare_derived(derived: &Derived<'_>, text: &str) -> Vec<Record> {
    let rel = derived.rel;
    let message = match derived.committed {
        None => format!(
            "{rel} is absent. The derived expiries are an artifact other gates read; \
             run `bash scripts/spec_conformance.sh --write` and commit it"
        ),
        Some(committed) if committed == text => return Vec::new(),
        Some(committed) => {
            let first = first_difference(&py_splitlines(committed), &py_splitlines(text));
            format!(
                "{rel} no longer matches the derivation from §12 (first difference at line \
                 {first}). A committed derivation that drifts from the table is the expiry \
                 nobody re-derived; run `bash scripts/spec_conformance.sh --write` and commit \
                 the result"
            )
        }
    };
    vec![Record::Violation(Violation {
        rule: "D5",
        row: "-".to_owned(),
        message,
    })]
}

/// `_first_difference`: the 1-based line where the two differ, or one past the shorter.
fn first_difference(have: &[&str], want: &[&str]) -> usize {
    have.iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .map_or(have.len().min(want.len()) + 1, |at| at + 1)
}

/// The CLI: the records as TSV lines and the exit code (1 when any VIOLATION, else 0).
/// `today` None reads `SPEC_CONFORMANCE_TODAY`, else the system date.
///
/// # Errors
///
/// The reason when an input cannot be read (exit 2 at the caller).
pub fn run_cli(
    spec_path: &Path,
    source: Option<String>,
    derived_path: Option<&Path>,
    today: Option<String>,
) -> Result<(String, u8), String> {
    let spec_bytes =
        std::fs::read(spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?;
    let spec = py_read_text(&spec_bytes);
    let today = match today {
        Some(day) => day,
        None => today_default()?,
    };
    let derived_file = derived_path
        .map(|p| read_optional(p).map(|committed| (py_path_str(p), committed)))
        .transpose()?;
    let derived = derived_file.as_ref().map(|(rel, committed)| Derived {
        rel: rel.as_str(),
        committed: committed.as_deref(),
    });
    let source = source.unwrap_or_else(|| py_path_str(spec_path));
    let records = check_dag(&spec, &source, derived, &today);
    let code = u8::from(records.iter().any(Record::is_violation));
    let mut out = String::new();
    for record in &records {
        out.push_str(&record.line());
        out.push('\n');
    }
    Ok((out, code))
}

/// The committed file's text, or None when it does not exist.
fn read_optional(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(py_read_text(&bytes))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// `_today()`: `SPEC_CONFORMANCE_TODAY` when it is set and not blank, else the local date.
fn today_default() -> Result<String, String> {
    if let Ok(value) = std::env::var("SPEC_CONFORMANCE_TODAY") {
        let value = value.trim();
        if !value.is_empty() {
            return Ok(value.to_owned());
        }
    }
    let out = Command::new("date")
        .arg("+%F")
        .output()
        .map_err(|e| format!("cannot run date (pass --today): {e}"))?;
    if !out.status.success() {
        return Err("date failed (pass --today)".to_owned());
    }
    String::from_utf8(out.stdout)
        .map(|s| s.trim().to_owned())
        .map_err(|e| format!("date output is not UTF-8: {e}"))
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
            "**LAB** (operator ruling, see Appendix D #7534; was **OPEN**). Expires **2026-11-27**";
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
    fn an_expired_lab_row_without_its_ruling_and_ticket_is_still_d6() {
        // C324: a LAB row has one ticket and one owner; the exemption is the ruling the cell
        // cites, so a bare LAB, or one missing either citation, is still an andon.
        for cell in [
            "**LAB**. Expires **2026-11-27**",
            "**LAB** (see Appendix D). Expires **2026-11-27**",
            "**LAB** (#7534). Expires **2026-11-27**",
            "**LAB** (see Appendix D #x). Expires **2026-11-27**",
        ] {
            let recs = check(&one_row(cell), TODAY);
            assert_eq!(
                violations(&recs),
                vec![("D6", "r1".to_owned())],
                "cell: {cell}"
            );
            assert!(lab_rows(&recs).is_empty(), "cell: {cell}");
        }
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
