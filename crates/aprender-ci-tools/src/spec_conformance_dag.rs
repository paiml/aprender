//! The §12 expiry DAG and its violation rules D0-D4 (PR1 of the port of
//! `scripts/lib/spec_conformance.py`: `parse_dag`, `_Expiries`, `derive_expiries`).
//!
//! Expiry is either a literal date on a root row or the latest expiry among its live
//! blockers; `expires` is a date only on root rows. The walk is the Python's, step for
//! step, so the violations come out in the same order with the same text.
//! Python `scripts/lib/spec_conformance.py` stays the reference until the parity lane
//! is wired (PR2+); this module is not yet called from `main.rs`.

use crate::pystr::{py_list_repr, py_repr};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// One violation, as the Python emits it: `VIOLATION\t<rule>\t<row>\t<message>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub rule: &'static str,
    pub row: String,
    pub message: String,
}

impl Violation {
    /// One TSV line, with tabs inside a field replaced by spaces, as `emit()` does.
    #[must_use]
    pub fn line(&self) -> String {
        let field = |s: &str| s.replace('\t', " ");
        format!(
            "VIOLATION\t{}\t{}\t{}",
            self.rule,
            field(&self.row),
            field(&self.message)
        )
    }
}

/// One §12 row as the walk sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub key: String,
    pub blocked_by: Vec<String>,
    pub expires_cell: String,
}

/// The rows, in table order, keyed by row id.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Dag {
    rows: Vec<Row>,
    index: HashMap<String, usize>,
}

impl Dag {
    /// A repeated key keeps its first position and takes the last row, as a Python dict does.
    pub fn new(rows: Vec<Row>) -> Self {
        let mut dag = Self::default();
        for row in rows {
            match dag.index.get(&row.key) {
                Some(&at) => dag.rows[at] = row,
                None => {
                    dag.index.insert(row.key.clone(), dag.rows.len());
                    dag.rows.push(row);
                }
            }
        }
        dag
    }

    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    fn position(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }
}

/// The derived expiry of one row and how it was derived.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Expiry {
    pub value: Option<String>,
    pub via: Vec<String>,
    pub discharged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Analysis {
    pub expiries: Vec<(String, Expiry)>,
    pub violations: Vec<Violation>,
}

/// Parse the first pipe table in `text` into a DAG (`parse_dag`).
///
/// # Errors
///
/// D0 when the table has no data rows, or no `blocked_by` / `expires` column.
pub fn from_markdown_table(text: &str) -> Result<Dag, Violation> {
    let lines: Vec<&str> = text.lines().collect();
    let (header, rows) = table_rows(&lines);
    if rows.is_empty() {
        return Err(violation(
            "D0",
            "-",
            "no §12 table parsed -- every non-root expiry is DERIVED from the \
             blocked_by column, and with no table there is nothing to derive from"
                .to_owned(),
        ));
    }
    let row_at = find(&header, "row")
        .or_else(|| find(&header, "id"))
        .unwrap_or(0);
    let (Some(blocked_at), Some(expires_at)) = (find(&header, "blocked"), find(&header, "expir"))
    else {
        return Err(violation(
            "D0",
            "-",
            format!(
                "§12 has no `blocked_by` and/or `expires` column (header: {})",
                py_list_repr(&header)
            ),
        ));
    };
    let mut order: Vec<String> = Vec::new();
    let mut raw: HashMap<String, (String, String)> = HashMap::new();
    for cells in &rows {
        let key = cell(cells, row_at);
        if key.is_empty() {
            continue;
        }
        if !raw.contains_key(&key) {
            order.push(key.clone());
        }
        raw.insert(key, (cell(cells, blocked_at), cell(cells, expires_at)));
    }
    // TWO PASSES, as in Python: a blocker is named by row id, and the id set is
    // only known once every row has been read.
    let built = order
        .iter()
        .map(|key| {
            let (text, expires) = &raw[key];
            Row {
                key: key.clone(),
                blocked_by: blockers_in_text(key, text, &order),
                expires_cell: expires.clone(),
            }
        })
        .collect();
    Ok(Dag::new(built))
}

/// The ids in `ids` (other than `key`) that `text` names as whole tokens: an id
/// is named only where neither neighbour is a word character or `-`, so `gx10`
/// and `row-2` do not name `10` or `2`.
#[must_use]
pub fn blockers_in_text(key: &str, text: &str, ids: &[String]) -> Vec<String> {
    ids.iter()
        .filter(|other| other.as_str() != key && names_whole_token(text, other))
        .cloned()
        .collect()
}

fn names_whole_token(text: &str, id: &str) -> bool {
    (0..=text.len())
        .filter(|&at| text.is_char_boundary(at))
        .any(|at| {
            text[at..].starts_with(id)
                && !text[..at].chars().next_back().is_some_and(is_token_char)
                && !text[at + id.len()..]
                    .chars()
                    .next()
                    .is_some_and(is_token_char)
        })
}

fn is_token_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

/// Derive every row's expiry and emit D1-D4 (`derive_expiries`).
#[must_use]
pub fn analyze(dag: &Dag) -> Analysis {
    let mut walk = Walk {
        dag,
        out: vec![None; dag.rows.len()],
        open: vec![false; dag.rows.len()],
        violations: Vec::new(),
    };
    for idx in 0..dag.rows.len() {
        walk.visit(idx, &[]);
    }
    let expiries = dag
        .rows
        .iter()
        .enumerate()
        .map(|(idx, row)| (row.key.clone(), walk.out[idx].clone().unwrap_or_default()))
        .collect();
    Analysis {
        expiries,
        violations: walk.violations,
    }
}

struct Walk<'a> {
    dag: &'a Dag,
    out: Vec<Option<Expiry>>,
    open: Vec<bool>,
    violations: Vec<Violation>,
}

impl Walk<'_> {
    fn visit(&mut self, idx: usize, stack: &[String]) -> Option<String> {
        if let Some(done) = &self.out[idx] {
            return done.value.clone();
        }
        if self.open[idx] {
            return self.cycle(idx, stack);
        }
        self.open[idx] = true;
        let dag = self.dag;
        let row = &dag.rows[idx];
        // A DISCHARGED row has no deadline to derive: the work landed.
        if row.expires_cell.to_uppercase().contains("LANDED") {
            return self.finish(idx, None, Vec::new(), true);
        }
        self.resolve(idx, stack)
    }

    fn cycle(&mut self, idx: usize, stack: &[String]) -> Option<String> {
        let dag = self.dag;
        let key = dag.rows[idx].key.as_str();
        let path: Vec<&str> = stack
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(key))
            .collect();
        self.violations.push(violation(
            "D1",
            key,
            format!(
                "blocked_by forms a CYCLE ({}). A cycle has no latest blocker, so \
                 every row in it would wait for itself",
                path.join(" -> ")
            ),
        ));
        self.out[idx] = Some(Expiry::default());
        None
    }

    fn resolve(&mut self, idx: usize, stack: &[String]) -> Option<String> {
        let dag = self.dag;
        let row = &dag.rows[idx];
        let literal = literal_of(&row.expires_cell);
        let live = self.live(row);
        if !row.blocked_by.is_empty() && live.is_empty() {
            return self.unblocked(idx, literal);
        }
        if row.blocked_by.is_empty() {
            return self.root(idx, literal, &row.expires_cell);
        }
        self.from_blockers(idx, stack, &live, literal)
    }

    fn live(&self, row: &Row) -> Vec<String> {
        row.blocked_by
            .iter()
            .filter(|b| {
                self.dag.position(b).is_some_and(|at| {
                    !self.dag.rows[at]
                        .expires_cell
                        .to_uppercase()
                        .contains("LANDED")
                })
            })
            .cloned()
            .collect()
    }

    fn unblocked(&mut self, idx: usize, literal: Option<String>) -> Option<String> {
        if literal.is_none() {
            let key = self.dag.rows[idx].key.clone();
            self.violations.push(violation(
                "D2",
                &key,
                "is blocked only by rows that have LANDED, so nothing derives its \
                 expiry any more, and it carries no date. An unblocked obligation \
                 with no deadline never expires"
                    .to_owned(),
            ));
        }
        self.finish(idx, literal, Vec::new(), false)
    }

    fn root(&mut self, idx: usize, literal: Option<String>, cell: &str) -> Option<String> {
        if literal.is_none() {
            let key = self.dag.rows[idx].key.clone();
            self.violations.push(violation(
                "D2",
                &key,
                format!(
                    "is a ROOT row (nothing blocks it) and carries no literal expiry \
                     {}. A root has nothing to derive from, so its date is the one \
                     date a person must write",
                    py_repr(cell)
                ),
            ));
        }
        self.finish(idx, literal, Vec::new(), false)
    }

    fn from_blockers(
        &mut self,
        idx: usize,
        stack: &[String],
        live: &[String],
        literal: Option<String>,
    ) -> Option<String> {
        let dag = self.dag;
        let row = &dag.rows[idx];
        if let Some(date) = &literal {
            self.violations.push(violation(
                "D3",
                &row.key,
                format!(
                    "is blocked by {} (still live: {}) and still types the literal \
                     date {}. §12's own preamble says `expires` is a date only on \
                     root rows; a typed expiry on a blocked row can fall BEFORE the \
                     work it waits on, which is how a gate comes to be red for a \
                     reason nobody can clear",
                    row.blocked_by.join(", "),
                    live.join(", "),
                    date
                ),
            ));
        }
        let (best, via) = self.best_blocker(idx, stack);
        self.finish(idx, best, via, false)
    }

    fn best_blocker(&mut self, idx: usize, stack: &[String]) -> (Option<String>, Vec<String>) {
        let dag = self.dag;
        let row = &dag.rows[idx];
        let mut child_stack = stack.to_vec();
        child_stack.push(row.key.clone());
        let mut best: Option<String> = None;
        let mut via: Vec<String> = Vec::new();
        for blocker in &row.blocked_by {
            let value = match dag.position(blocker) {
                Some(child) => self.visit(child, &child_stack),
                None => {
                    self.violations.push(violation(
                        "D4",
                        &row.key,
                        format!(
                            "is blocked_by {}, which is not a row in §12",
                            py_repr(blocker)
                        ),
                    ));
                    None
                }
            };
            let Some(value) = value else { continue };
            let ordering = best.as_deref().map(|current| value.as_str().cmp(current));
            match ordering {
                Some(std::cmp::Ordering::Less) => {}
                Some(std::cmp::Ordering::Equal) => via.push(blocker.clone()),
                _ => {
                    best = Some(value);
                    via = vec![blocker.clone()];
                }
            }
        }
        (best, via)
    }

    fn finish(
        &mut self,
        idx: usize,
        value: Option<String>,
        via: Vec<String>,
        discharged: bool,
    ) -> Option<String> {
        self.out[idx] = Some(Expiry {
            value: value.clone(),
            via,
            discharged,
        });
        self.open[idx] = false;
        value
    }
}

fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"Expires\s+(\d{4}-\d{2}-\d{2})").expect("static pattern"))
}

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d{4}-\d{2}-\d{2}").expect("static pattern"))
}

/// The expiry a cell states: the `Expires <date>` marker wins, else the first date.
fn literal_of(cell: &str) -> Option<String> {
    if let Some(caps) = marker_re().captures(cell) {
        return Some(caps[1].to_owned());
    }
    date_re().find(cell).map(|m| m.as_str().to_owned())
}

fn violation(rule: &'static str, row: &str, message: String) -> Violation {
    Violation {
        rule,
        row: row.to_owned(),
        message,
    }
}

fn strip_md(cell: &str) -> String {
    cell.replace("**", "").replace('*', "").trim().to_owned()
}

/// `_table_cells`: None for a non-table line, `[]` for a separator, else the cells.
fn table_cells(line: &str) -> Option<Vec<String>> {
    if !line.trim_start().starts_with('|') {
        return None;
    }
    let cells: Vec<String> = line
        .trim()
        .trim_matches('|')
        .split('|')
        .map(strip_md)
        .collect();
    if cells.concat().chars().all(|c| "-: ".contains(c)) {
        return Some(Vec::new());
    }
    Some(cells)
}

/// `table_rows`: (lower-cased header, data rows) of the first pipe table.
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

fn find(header: &[String], needle: &str) -> Option<usize> {
    header.iter().position(|name| name.contains(needle))
}

fn cell(cells: &[String], at: usize) -> String {
    cells
        .get(at)
        .map_or_else(String::new, |c| c.trim().trim_matches('`').to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const D1: &str = include_str!("../tests/fixtures/spec_dag/d1_cycle.md");
    const D2: &str = include_str!("../tests/fixtures/spec_dag/d2_no_date.md");
    const D3: &str = include_str!("../tests/fixtures/spec_dag/d3_typed_on_blocked.md");

    fn fired(a: &Analysis) -> Vec<(&'static str, String)> {
        a.violations
            .iter()
            .map(|v| (v.rule, v.row.clone()))
            .collect()
    }

    fn check(text: &str) -> Analysis {
        let dag = from_markdown_table(text).expect("fixture parses");
        analyze(&dag)
    }

    fn expiry_of(a: &Analysis, key: &str) -> Option<String> {
        a.expiries
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, e)| e.value.clone())
    }

    #[test]
    fn d1_cycle_in_blocked_by_fires_once_and_names_the_loop() {
        let a = check(D1);
        assert_eq!(fired(&a), vec![("D1", "r1".to_owned())]);
        assert!(
            a.violations[0].message.contains("r1 -> r2 -> r1"),
            "{}",
            a.violations[0].message
        );
    }

    #[test]
    fn d2_fires_on_a_root_and_on_an_unblocked_row_with_no_literal_date() {
        let a = check(D2);
        assert_eq!(
            fired(&a),
            vec![("D2", "r1".to_owned()), ("D2", "r3".to_owned())]
        );
    }

    #[test]
    fn d3_fires_on_a_blocked_row_that_types_a_literal_date() {
        let a = check(D3);
        assert_eq!(fired(&a), vec![("D3", "r2".to_owned())]);
    }

    #[test]
    fn d4_fires_on_a_blocker_that_is_not_a_row() {
        let dag = Dag::new(vec![
            Row {
                key: "r1".into(),
                blocked_by: vec!["ghost".into(), "r2".into()],
                expires_cell: "derived".into(),
            },
            Row {
                key: "r2".into(),
                blocked_by: vec![],
                expires_cell: "2026-02-01".into(),
            },
        ]);
        let a = analyze(&dag);
        assert_eq!(fired(&a), vec![("D4", "r1".to_owned())]);
        assert!(a.violations[0].message.contains("ghost"));
        assert_eq!(expiry_of(&a, "r1").as_deref(), Some("2026-02-01"));
    }

    #[test]
    fn a_landed_blocker_is_not_live_so_its_dependent_is_unblocked_not_d3() {
        let text = "| row | blocked_by | expires |\n|---|---|---|\n\
                    | r1 | — | **LANDED** (this PR) |\n\
                    | r2 | r1 | 2025-01-01 |\n";
        let a = check(text);
        assert_eq!(fired(&a), Vec::new());
        assert_eq!(expiry_of(&a, "r2").as_deref(), Some("2025-01-01"));
    }

    #[test]
    fn a_derived_expiry_is_the_latest_live_blocker() {
        let text = "| row | blocked_by | expires |\n|---|---|---|\n\
                    | r1 | — | 2026-01-01 |\n\
                    | r2 | — | 2026-03-01 |\n\
                    | r3 | r1, r2 | derived |\n";
        let a = check(text);
        assert_eq!(fired(&a), Vec::new());
        assert_eq!(expiry_of(&a, "r3").as_deref(), Some("2026-03-01"));
    }

    #[test]
    fn an_expires_marker_beats_an_earlier_prose_date() {
        let text = "| row | blocked_by | expires |\n|---|---|---|\n\
                    | r1 | — | witness 2026-09-02, Expires 2026-09-19 |\n";
        let a = check(text);
        assert_eq!(expiry_of(&a, "r1").as_deref(), Some("2026-09-19"));
    }

    #[test]
    fn blockers_are_whole_ids_not_substrings_or_hyphen_joined() {
        let ids: Vec<String> = ["15", "10", "2", "r1"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let text = "15 clean at `--phase merge`, gx10 window, 115 ms, row-2";
        assert_eq!(blockers_in_text("r1", text, &ids), vec!["15".to_owned()]);
        assert_eq!(blockers_in_text("15", "15", &ids), Vec::<String>::new());
    }

    #[test]
    fn a_table_without_blocked_by_and_expires_columns_is_d0() {
        let err = from_markdown_table("| a | b |\n|---|---|\n| x | y |\n")
            .expect_err("no blocked_by column");
        assert_eq!(err.rule, "D0");
    }
}
