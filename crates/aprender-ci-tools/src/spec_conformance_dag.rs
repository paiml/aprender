//! The §12 expiry DAG and its violation rules D0-D4 (PR1 of the port of
//! `scripts/lib/spec_conformance.py`: `parse_dag`, `_Expiries`, `derive_expiries`).
//!
//! STUB: RED commit. The API is final; the behaviour is not yet written.

use std::collections::HashMap;

/// One violation, as the Python emits it: `VIOLATION\t<rule>\t<row>\t<message>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub rule: &'static str,
    pub row: String,
    pub message: String,
}

impl Violation {
    pub fn line(&self) -> String {
        format!("VIOLATION\t{}\t{}\t{}", self.rule, self.row, self.message)
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

    pub fn rows(&self) -> &[Row] {
        &self.rows
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

/// Parse the §12 pipe table (first table in `text`) into a DAG. D0 on a missing table or columns.
pub fn from_markdown_table(_text: &str) -> Result<Dag, Violation> {
    Ok(Dag::default())
}

/// The ids in `ids` (other than `key`) that `text` names as whole tokens.
pub fn blockers_in_text(_key: &str, _text: &str, _ids: &[String]) -> Vec<String> {
    Vec::new()
}

/// Derive every row's expiry and emit D1-D4.
pub fn analyze(_dag: &Dag) -> Analysis {
    Analysis::default()
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
