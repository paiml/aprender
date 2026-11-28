//! L0-L3 of the PP-9 ledger (PMAT-7534; the port of `check_ledger` and its helpers in
//! `scripts/lib/spec_conformance.py`, with the selftest's `LEDGER_*` fixtures as unit tests).
//!
//! PP-9: a cell, once run, is SPENT. A re-run that comes out green is the defect, and it is
//! refusable only against the written ledger. L0 is the ledger that cannot be read as a
//! ledger, L1 a re-spent key, L2 a spend row outside the table PP-9 reads, L3 a malformed row.

use crate::pystr::{is_py_space, py_strip, py_text_lines};
use crate::spec_conformance_dag::Violation;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// `SPEND_TIERS`: a cell that starts with one of these, case-insensitively, is a spend claim.
const SPEND_TIERS: [&str; 2] = ["RECORDED", "CONFORMANT"];

/// `_WRAP`: the characters `_norm` strips from both ends of a value.
const WRAP: [char; 5] = ['`', '*', '_', ' ', '\t'];

/// The `Cf` (format) code points as inclusive ranges, from `unicodedata.category` at Unicode
/// 13.0.0, the table the `python3` on the CI box reports. The Rust standard library has no
/// general-category lookup and the crate takes no dependency for one, so the table is written
/// out here; a newer Unicode adds ranges (U+0890..U+0891 is Unicode 14) that this list lacks.
const CF_RANGES: [(u32, u32); 20] = [
    (0xAD, 0xAD),
    (0x600, 0x605),
    (0x61C, 0x61C),
    (0x6DD, 0x6DD),
    (0x70F, 0x70F),
    (0x8E2, 0x8E2),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x206F),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x13430, 0x13438),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001),
    (0xE0020, 0xE007F),
];

const NOTE_TEXT: &str = "no ledger row is marked conformance RECORDED, so the PP-9 duplicate \
    rule matched nothing on this tree; its must-fire lives in the fixture rows of --selftest";

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
    let Some(text) = text else {
        return vec![violation(
            "L0",
            "-",
            format!(
                "{path_label} is absent -- PP-9 ('a cell, once run, is spent') is only \
                 enforceable against a written record of what has been spent"
            ),
        )];
    };
    let owned = py_text_lines(text.as_bytes());
    let lines: Vec<&str> = owned.iter().map(|l| l.trim_end_matches('\n')).collect();
    let (header, rows) = table_rows(&lines);
    if rows.is_empty() {
        return vec![violation(
            "L0",
            "-",
            "no row table parsed from the ledger".to_owned(),
        )];
    }
    let mut out = Vec::new();
    let universe = ledger_universe(&lines, &header, &rows, &mut out);
    let cols = column_index(&header, &mut out);
    let recorded = check_spends(&universe, &cols, &mut out);
    out.push(Record::Ledger {
        rows: universe.len(),
        recorded,
    });
    if recorded == 0 {
        out.push(Record::Note(NOTE_TEXT.to_owned()));
    }
    out
}

fn violation(rule: &'static str, row: &str, message: String) -> Record {
    Record::Violation(Violation {
        rule,
        row: row.to_owned(),
        message,
    })
}

// --------------------------------------------------------------- markdown ---

fn code_tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)</?code>").expect("static pattern"))
}

fn row_id_cell_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[0-9]+[a-z]?$").expect("static pattern"))
}

/// `strip_md`: `**` and `*` come off, then the whitespace.
fn strip_md(cell: &str) -> String {
    py_strip(&cell.replace("**", "").replace('*', "")).to_owned()
}

/// `set("".join(cells)) <= set("-: ")`: the cells are a separator row's cells (or empty).
fn is_separator_chars(cells: &[String]) -> bool {
    cells.concat().chars().all(|c| "-: ".contains(c))
}

/// `_table_cells`: None for a line that is not a table row, `[]` for a separator row, else the
/// row's cells.
fn table_cells(line: &str) -> Option<Vec<String>> {
    if !line.trim_start_matches(is_py_space).starts_with('|') {
        return None;
    }
    let cells: Vec<String> = py_strip(line)
        .trim_matches('|')
        .split('|')
        .map(strip_md)
        .collect();
    if is_separator_chars(&cells) {
        return Some(Vec::new());
    }
    Some(cells)
}

/// `table_rows`: (lower-cased header, data rows) of the first pipe table in `lines`.
fn table_rows(lines: &[&str]) -> (Vec<String>, Vec<Vec<String>>) {
    let mut header: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        match table_cells(line) {
            None if !rows.is_empty() => break,
            None => {}
            Some(cells) if cells.is_empty() => {}
            Some(cells) if header.is_empty() => {
                header = cells.iter().map(|c| c.to_lowercase()).collect();
            }
            Some(cells) => rows.push(cells),
        }
    }
    (header, rows)
}

/// `column`: the index of the first header cell containing every needle, or None.
fn column(header: &[String], needles: &[&str]) -> Option<usize> {
    header
        .iter()
        .position(|name| needles.iter().all(|n| name.contains(n)))
}

/// `_first_table_end`: the line index at which `table_rows` stops, or `lines.len()`.
fn first_table_end(lines: &[&str]) -> usize {
    let mut header_seen = false;
    let mut rows_seen = false;
    for (i, line) in lines.iter().enumerate() {
        if !line.trim_start_matches(is_py_space).starts_with('|') {
            if rows_seen {
                return i;
            }
            continue;
        }
        let cells: Vec<String> = py_strip(line)
            .trim_matches('|')
            .split('|')
            .map(strip_md)
            .collect();
        if is_separator_chars(&cells) {
            continue;
        }
        if header_seen {
            rows_seen = true;
        } else {
            header_seen = true;
        }
    }
    lines.len()
}

/// `_pipe_cells`: the cells of a pipe line whose leading and trailing pipes are optional; `[]`
/// when the line holds fewer than two pipes.
fn pipe_cells(line: &str) -> Vec<String> {
    if line.matches('|').count() < 2 {
        return Vec::new();
    }
    let body = py_strip(line);
    let body = body.strip_prefix('|').unwrap_or(body);
    let body = body.strip_suffix('|').unwrap_or(body);
    body.split('|').map(strip_md).collect()
}

/// `_is_separator`.
fn is_separator(cells: &[String]) -> bool {
    !cells.is_empty() && is_separator_chars(cells)
}

// ------------------------------------------------------------ normalisation ---

/// `_norm`: the value a person meant. Code tags, `Cf` format characters, and the whitespace
/// runs come out, the result is trimmed of the wrap characters, and nothing else changes.
fn norm(cell: &str) -> String {
    let without_tags = code_tag_re().replace_all(cell, "");
    let visible: String = without_tags.chars().filter(|c| !is_cf(*c)).collect();
    let words: Vec<&str> = visible
        .split(is_py_space)
        .filter(|w| !w.is_empty())
        .collect();
    words.join(" ").trim_matches(WRAP).to_owned()
}

fn is_cf(c: char) -> bool {
    let cp = u32::from(c);
    CF_RANGES.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp))
}

/// The comparison form of a key: `str.casefold()` is approximated by lower, upper, lower, which
/// agrees on every fold the ledger's keys are written with (ASCII, `ß`, the final sigma).
fn casefold(s: &str) -> String {
    s.to_lowercase().to_uppercase().to_lowercase()
}

fn non_empty(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

/// `_row_id`: the first cell as a ledger row id (`N` or `Na`), or "".
fn row_id(cells: &[String]) -> String {
    let Some(first) = cells.first() else {
        return String::new();
    };
    let rid = norm(first);
    if row_id_cell_re().is_match(&rid) {
        rid
    } else {
        String::new()
    }
}

/// A value that starts with RECORDED or CONFORMANT, case-insensitively, is a spend.
fn spend_tier(text: &str) -> bool {
    let upper = text.to_uppercase();
    SPEND_TIERS.iter().any(|tier| upper.starts_with(tier))
}

/// `_claims_spend`: any cell of the row starts with a tier.
fn claims_spend(cells: &[String]) -> bool {
    cells.iter().any(|c| spend_tier(&norm(c)))
}

// ---------------------------------------------------------- L2 and L3 shapes ---

/// A spend row after the first table: its line index, its reported id, and its cells.
struct Outside {
    line: usize,
    rid: String,
    cells: Vec<String>,
}

/// `_outside_row`: (id or the raw first cell or "?", cells) when the line is a pipe row that
/// claims a spend, else None.
fn outside_row(line: &str) -> Option<(String, Vec<String>)> {
    let cells = pipe_cells(line);
    if cells.is_empty() || is_separator(&cells) || !claims_spend(&cells) {
        return None;
    }
    let rid = non_empty(row_id(&cells))
        .or_else(|| cells.first().and_then(|c| non_empty(norm(c))))
        .unwrap_or_else(|| "?".to_owned());
    Some((rid, cells))
}

/// `ledger_rows_outside_table`: every spend row from the line that ends the first table onward.
fn outside_rows(lines: &[&str], table_end: usize) -> Vec<Outside> {
    lines
        .iter()
        .enumerate()
        .skip(table_end)
        .filter_map(|(line, text)| {
            outside_row(text).map(|(rid, cells)| Outside { line, rid, cells })
        })
        .collect()
}

/// `_emit_ledger_split`: L2, one record for every spend row outside the first table.
fn split_record(table_end: usize, outside: &[Outside]) -> Record {
    let ids: Vec<&str> = outside.iter().map(|o| o.rid.as_str()).collect();
    let first_line = outside.first().map_or(0, |o| o.line);
    violation(
        "L2",
        &ids.join(" "),
        format!(
            "{} ledger row(s) sit outside the table PP-9 reads: the table ends at line {} (a \
             blank or prose line splits it) and a row with the same columns continues at line \
             {}; every spent row must be contiguous with the header or the re-spend check \
             never sees it",
            outside.len(),
            table_end + 1,
            first_line + 1
        ),
    )
}

/// `_emit_malformed` + `_check_ledger_shapes`: L3 for every first-table row whose cell count
/// differs from the header's.
fn check_shapes(rows: &[Vec<String>], header_len: usize) -> Vec<Record> {
    rows.iter()
        .filter(|cells| cells.len() != header_len)
        .map(|cells| {
            let rid = non_empty(row_id(cells))
                .or_else(|| cells.first().cloned())
                .unwrap_or_else(|| "?".to_owned());
            violation(
                "L3",
                &rid,
                format!(
                    "ledger row {rid} has {} cell(s) against a {header_len}-cell header: a \
                     malformed row shifts every column the spend key reads, so it is refused \
                     rather than mis-keyed",
                    cells.len()
                ),
            )
        })
        .collect()
}

/// `_ledger_universe`: the first table's rows plus every spend row outside it. L2 and L3 are
/// emitted here, in that order.
fn ledger_universe(
    lines: &[&str],
    header: &[String],
    rows: &[Vec<String>],
    out: &mut Vec<Record>,
) -> Vec<Vec<String>> {
    let table_end = first_table_end(lines);
    let outside = outside_rows(lines, table_end);
    if !outside.is_empty() {
        out.push(split_record(table_end, &outside));
    }
    out.extend(check_shapes(rows, header.len()));
    rows.iter()
        .cloned()
        .chain(outside.into_iter().map(|o| o.cells))
        .collect()
}

// ------------------------------------------------------------ L0 and L1 ---

/// The indexes of the six columns the spend key and the tier are read from.
#[derive(Debug, Clone, Copy)]
struct Columns {
    host: Option<usize>,
    workload: Option<usize>,
    model: Option<usize>,
    commit: Option<usize>,
    interleaved: Option<usize>,
    conformance: Option<usize>,
}

impl Columns {
    /// The columns by name, in alphabetical order (the order the L0 message lists them).
    fn named(&self) -> [(&'static str, Option<usize>); 6] {
        [
            ("commit", self.commit),
            ("conformance", self.conformance),
            ("host", self.host),
            ("interleaved", self.interleaved),
            ("model", self.model),
            ("workload", self.workload),
        ]
    }
}

/// `_ledger_column_index`: the six columns, and L0 for every one the first table lacks.
fn column_index(header: &[String], out: &mut Vec<Record>) -> Columns {
    let cols = Columns {
        host: column(header, &["host"]),
        workload: column(header, &["workload"]),
        model: column(header, &["model"]),
        commit: column(header, &["commit"]),
        interleaved: column(header, &["interleav"]),
        conformance: column(header, &["conformance"]),
    };
    let missing: Vec<&str> = cols
        .named()
        .iter()
        .filter(|(_, idx)| idx.is_none())
        .map(|(name, _)| *name)
        .collect();
    if !missing.is_empty() {
        out.push(violation(
            "L0",
            "-",
            format!(
                "the first pipe table of the ledger carries no {} column: it is not the table \
                 PP-9 reads, so the spend check has nothing to key on -- a table written above \
                 the ledger shadows it",
                missing.join(", ")
            ),
        ));
    }
    cols
}

/// `_ledger_cell`: the normalised value at `idx`, or "" when there is no such column or cell.
fn cell(cells: &[String], idx: Option<usize>) -> String {
    idx.and_then(|i| cells.get(i))
        .map(|c| norm(c))
        .unwrap_or_default()
}

/// `_ledger_spend_key`: (host, workload, model, commit, interleaved), compared case-folded.
fn spend_key(cells: &[String], cols: &Columns) -> Vec<String> {
    [
        cols.host,
        cols.workload,
        cols.model,
        cols.commit,
        cols.interleaved,
    ]
    .iter()
    .map(|idx| casefold(&cell(cells, *idx)))
    .collect()
}

/// `_emit_respend`: L1 for a key that a second RECORDED row shares.
fn respend_record(key: &[String]) -> Record {
    let row: Vec<&str> = key
        .iter()
        .filter(|k| !k.is_empty())
        .map(String::as_str)
        .collect();
    violation(
        "L1",
        &row.join(" "),
        "two ledger rows share the spend key (host, workload, model quant, commit, \
         interleaved) with conformance RECORDED. PP-9: a cell, once run, is SPENT -- the \
         second run is a re-roll, and the only legal move is a new commit"
            .to_owned(),
    )
}

/// `_check_ledger_spends`: L1 for every re-spent key among the RECORDED rows; returns how many
/// rows are RECORDED.
fn check_spends(rows: &[Vec<String>], cols: &Columns, out: &mut Vec<Record>) -> usize {
    let mut recorded = 0;
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    for cells in rows {
        if !spend_tier(&cell(cells, cols.conformance)) {
            continue;
        }
        recorded += 1;
        let key = spend_key(cells, cols);
        if !seen.insert(key.clone()) {
            out.push(respend_record(&key));
        }
    }
    recorded
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
