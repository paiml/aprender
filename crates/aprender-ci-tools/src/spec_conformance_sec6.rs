//! §6 of the PP-LLAMA-001 master: rules C0, C1, C2, C3 and C5, and the case surfaces they join
//! to (PMAT-7534; the port of `section_lines`, `selftest_names`, `Surfaces`, `_check_case`,
//! `check_section_6` and `check_id_contiguity` in `scripts/lib/spec_conformance.py`).
//!
//! The Python keeps `SURFACE_TOKEN` as one regex with a lookbehind and a lookahead; the
//! workspace `regex` has neither, so the token scanner here checks the neighbouring characters
//! by hand. The surfaces are injected through [`Surfaces`], so the rules run without a
//! subprocess; [`Tree`] is the real resolver (`perf_gate.sh`, a script's own case table, and the
//! `#[test]` functions under `crates/<crate>/src`).

use crate::spec_conformance_dag::Violation;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// One line of §6's output: a `ROW`, a `CASE`, or a violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    /// `ROW\t<status or (empty)>\t<key>`.
    Row {
        status: String,
        key: String,
    },
    /// `CASE\t<surface>\t<name>\t<found|missing>\t<key>`.
    Case {
        surface: String,
        name: String,
        found: bool,
        key: String,
    },
    Violation(Violation),
}

impl Record {
    /// The TSV line the Python `emit()` writes for this record.
    #[must_use]
    pub fn line(&self) -> String {
        match self {
            Self::Row { status, key } => tsv(&["ROW", status, key]),
            Self::Case {
                surface,
                name,
                found,
                key,
            } => tsv(&[
                "CASE",
                surface,
                name,
                if *found { "found" } else { "missing" },
                key,
            ]),
            Self::Violation(v) => v.line(),
        }
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

/// What §6 produced: the number of rows parsed, and the records in emit order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Section6 {
    pub rows: usize,
    pub records: Vec<Record>,
}

/// Every case name a surface can produce. `prefix` is `pg`, `sh:<script>` or `rs:<crate>`;
/// None means this tree cannot enumerate the surface (C3).
pub trait Surfaces {
    fn names(&self, prefix: &str) -> Option<HashSet<String>>;
}

/// The real resolver, rooted at a repository. Each surface is resolved once and cached.
pub struct Tree {
    root: PathBuf,
    cache: RefCell<HashMap<String, Option<HashSet<String>>>>,
}

impl Tree {
    #[must_use]
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            cache: RefCell::new(HashMap::new()),
        }
    }
}

impl Surfaces for Tree {
    fn names(&self, prefix: &str) -> Option<HashSet<String>> {
        let _ = (&self.root, &self.cache);
        let _ = prefix;
        None
    }
}

/// `SURFACE_TOKEN` at byte offset `at` of `text`: the end offset of the match, or None.
fn surface_end(text: &str, at: usize) -> Option<usize> {
    let _ = (text, at);
    None
}

/// `selftest_names`: (surface, name) for every selftest case a §6 row names.
#[must_use]
pub fn selftest_names(cell: &str) -> Vec<(String, String)> {
    let _ = cell;
    Vec::new()
}

/// The names of a `--list-selftests` reply: rc 0 and every non-empty line a bare identifier.
#[must_use]
pub fn list_mode_names(rc_ok: bool, stdout: &str) -> Option<HashSet<String>> {
    let _ = (rc_ok, stdout);
    None
}

/// The names on a case table's own `ok` / `BROKE` lines.
#[must_use]
pub fn case_names(text: &str) -> HashSet<String> {
    let _ = text;
    HashSet::new()
}

/// The `#[test]` function names in one Rust file's lines.
#[must_use]
pub fn test_fn_names(lines: &[String]) -> HashSet<String> {
    let _ = lines;
    HashSet::new()
}

/// `section_lines` for §6: the lines under the first §6 heading, up to the next heading.
#[must_use]
pub fn section_lines(text: &str) -> Vec<&str> {
    let _ = text;
    Vec::new()
}

/// `check_section_6`: rules C0, C1, C2 and C3 over the §6 table of `spec_text`.
/// `spec_rel` is the path the C0 message names.
#[must_use]
pub fn check_section_6(spec_rel: &str, spec_text: &str, surfaces: &dyn Surfaces) -> Section6 {
    let _ = (spec_rel, spec_text, surfaces);
    Section6::default()
}

/// `check_id_contiguity`: rule C5 over the `PP-nn` ids of `spec_text`.
#[must_use]
pub fn check_id_contiguity(spec_text: &str) -> Vec<Record> {
    let _ = spec_text;
    Vec::new()
}

/// The §6 and C5 records of one spec, in the Python's order, resolved against `root`.
///
/// # Errors
///
/// The reason the spec cannot be read.
pub fn scan(root: &Path, spec: &Path) -> Result<Vec<Record>, String> {
    let _ = (root, spec);
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC_REL: &str = "docs/specifications/PP-LLAMA-001-MASTER.md";
    const ARMED: &str =
        "| PP-1 | a rule | mutate | hold | ARMED | `scripts/x.sh` · `pg:alpha` / `pg:beta` |";
    const OPEN: &str =
        "| PP-2 | a rule | mutate | hold | OPEN | `scripts/x.sh` · `pg:gamma` / `pg:delta` |";
    const PP3: &str =
        "| PP-3 | a rule | mutate | hold | ARMED | `scripts/x.sh` · `pg:alpha` / `pg:beta` |";
    const FIXTURE_TREE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/spec_sec6/rs_tree"
    );

    /// A surface table: each prefix with its names, or None for a surface that is absent.
    struct Fake(Vec<(&'static str, Option<Vec<&'static str>>)>);

    impl Surfaces for Fake {
        fn names(&self, prefix: &str) -> Option<HashSet<String>> {
            self.0
                .iter()
                .find(|(p, _)| *p == prefix)
                .and_then(|(_, names)| {
                    names
                        .as_ref()
                        .map(|v| v.iter().map(|s| (*s).to_owned()).collect())
                })
        }
    }

    fn pg(names: &[&'static str]) -> Fake {
        Fake(vec![("pg", Some(names.to_vec()))])
    }

    fn spec6(rows: &str) -> String {
        format!(
            "# PP-LLAMA-001 (fixture)\n\n## §6 Invariants\n\n\
             | id | rule | must-fire | must-not-fire | status | producer · selftest |\n\
             |---|---|---|---|---|---|\n{rows}\n\n## §12 Owed work\n\n\
             | row | what is owed | owner | blocked_by | expires |\n|---|---|---|---|---|\n"
        )
    }

    fn section6(rows: &str, surfaces: &dyn Surfaces) -> Section6 {
        check_section_6(SPEC_REL, &spec6(rows), surfaces)
    }

    /// The rules that fired, sorted and de-duplicated, comma-joined; CLEAN when none did.
    fn verdict(records: &[Record]) -> String {
        let mut rules: Vec<&str> = records
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

    fn lines(records: &[Record]) -> Vec<String> {
        records.iter().map(Record::line).collect()
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect()
    }

    // ---- the §6 cases of the bash selftest, same names --------------------------------

    #[test]
    fn conformance_ok() {
        let s = section6(ARMED, &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 1);
        assert_eq!(verdict(&s.records), "CLEAN");
    }

    #[test]
    fn conformance_missing() {
        let s = section6(ARMED, &pg(&["alpha"]));
        assert_eq!(s.rows, 1);
        assert_eq!(verdict(&s.records), "C1");
    }

    #[test]
    fn conformance_vacuity() {
        let s = section6("", &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 0);
        assert_eq!(verdict(&s.records), "C0");
    }

    #[test]
    fn conformance_open_row_ignored() {
        let s = section6(&format!("{ARMED}\n{OPEN}"), &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 2);
        assert_eq!(verdict(&s.records), "CLEAN");
    }

    #[test]
    fn conformance_gap_is_red() {
        let rows = format!("{ARMED}\n{PP3}");
        let mut records = section6(&rows, &pg(&["alpha", "beta"])).records;
        records.extend(check_id_contiguity(&spec6(&rows)));
        assert_eq!(verdict(&records), "C5");
    }

    // ---- must-fire cases the selftest never fires ----------------------------------

    #[test]
    fn c2_armed_row_naming_one_case() {
        let row = "| PP-1 | r | m | h | ARMED | `scripts/x.sh` · `pg:alpha` |";
        let s = section6(row, &pg(&["alpha", "beta"]));
        assert_eq!(verdict(&s.records), "C2");
        assert!(!lines(&s.records).iter().any(|l| l.starts_with("CASE")));
    }

    #[test]
    fn c3_absent_surface_is_loud_for_every_case() {
        let s = section6(ARMED, &Fake(Vec::new()));
        assert_eq!(verdict(&s.records), "C3");
        let c3 = s
            .records
            .iter()
            .filter(|r| matches!(r, Record::Violation(v) if v.rule == "C3"))
            .count();
        assert_eq!(c3, 2);
    }

    // ---- the alias convention: a short name is satisfied by its `__` long form -----

    #[test]
    fn alias_short_name_is_satisfied_by_its_long_form() {
        let row = "| PP-32 | r | m | h | ARMED | `rs:aprender-test-lib` · \
                   `rs:aprender-test-lib` `abrecord_comparator`, `abrecord_ok` |";
        let surf = Fake(vec![(
            "rs:aprender-test-lib",
            Some(vec![
                "abrecord_comparator__a_comparator_field_does_not_parse",
                "abrecord_ok__a_code_delta_with_two_shas_parses",
            ]),
        )]);
        let s = section6(row, &surf);
        assert_eq!(s.rows, 1);
        assert_eq!(verdict(&s.records), "CLEAN");
    }

    #[test]
    fn alias_does_not_launder_a_missing_case() {
        let row = "| PP-32 | r | m | h | ARMED | `rs:aprender-test-lib` `abrecord_comparator`, `abrecord_ok` |";
        let surf = Fake(vec![(
            "rs:aprender-test-lib",
            Some(vec!["abrecord_ok__only_this_one_exists"]),
        )]);
        let s = section6(row, &surf);
        assert_eq!(verdict(&s.records), "C1");
    }

    // ---- record order and shape --------------------------------------------------------

    #[test]
    fn records_follow_python_order() {
        let s = section6(ARMED, &pg(&["alpha", "beta"]));
        assert_eq!(
            lines(&s.records),
            vec![
                "ROW\tARMED\tPP-1",
                "CASE\tpg\talpha\tfound\tPP-1",
                "CASE\tpg\tbeta\tfound\tPP-1",
            ]
        );
    }

    #[test]
    fn missing_case_is_a_case_line_then_c1() {
        let s = section6(ARMED, &pg(&["alpha"]));
        let out = lines(&s.records);
        assert_eq!(out[2], "CASE\tpg\tbeta\tmissing\tPP-1");
        assert!(out[3].starts_with("VIOLATION\tC1\tPP-1\t"), "{}", out[3]);
    }

    #[test]
    fn non_armed_row_emits_its_row_and_no_cases() {
        let s = section6(OPEN, &pg(&[]));
        assert_eq!(lines(&s.records), vec!["ROW\tOPEN\tPP-2"]);
    }

    #[test]
    fn empty_status_is_reported_as_empty() {
        let s = section6("| PP-9 | r | m | h |  | `pg:x` / `pg:y` |", &pg(&[]));
        assert_eq!(lines(&s.records)[0], "ROW\t(empty)\tPP-9");
    }

    // ---- selftest_names: the two spellings, the inner rebind, the producer drop ---------

    #[test]
    fn selftest_names_prefixed_form() {
        assert_eq!(
            selftest_names("`sh:scripts/check_llama_pin.sh:pin_stale`"),
            pairs(&[("sh:scripts/check_llama_pin.sh", "pin_stale")])
        );
    }

    #[test]
    fn selftest_names_annotated_form_defaults_to_pg() {
        assert_eq!(
            selftest_names("`cellset_missing` / `cellset_na_ok` (pg)"),
            pairs(&[("pg", "cellset_missing"), ("pg", "cellset_na_ok")])
        );
    }

    #[test]
    fn selftest_names_inner_surface_rebinds_for_the_names_after_it() {
        assert_eq!(
            selftest_names("`join_mismatch` / `join_ok` (pg + rs:aprender-test-lib `a`, `b`)"),
            pairs(&[
                ("pg", "join_mismatch"),
                ("pg", "join_ok"),
                ("rs:aprender-test-lib", "a"),
                ("rs:aprender-test-lib", "b"),
            ])
        );
    }

    #[test]
    fn selftest_names_drops_the_producer_half() {
        assert_eq!(
            selftest_names("`scripts/x.sh` · `pg:alpha`"),
            pairs(&[("pg", "alpha")])
        );
    }

    // ---- the lookaround-free surface scanner ---------------------------------------------

    #[test]
    fn surface_token_needs_a_word_boundary_on_both_sides() {
        assert_eq!(surface_end("pg", 0), Some(2));
        assert_eq!(surface_end("xpg", 1), None);
        assert_eq!(surface_end("pg.", 0), None);
        assert_eq!(surface_end("pgx", 0), None);
        assert_eq!(surface_end("rs:crate-a b", 0), Some(10));
        assert_eq!(surface_end("sh:", 0), None);
    }

    // ---- surface readers --------------------------------------------------------------

    #[test]
    fn case_names_read_ok_and_broke_lines_only() {
        let text = "  ok    alpha  fine\nBROKE beta expected x\nnoise line\nokay nope\n";
        let want: HashSet<String> = ["alpha", "beta"].iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(case_names(text), want);
    }

    #[test]
    fn list_mode_needs_rc_zero_and_identifier_lines() {
        let names = |v: &[&str]| -> HashSet<String> { v.iter().map(|s| (*s).to_owned()).collect() };
        assert_eq!(
            list_mode_names(true, "alpha\nbeta\n"),
            Some(names(&["alpha", "beta"]))
        );
        assert_eq!(list_mode_names(false, "alpha\n"), None);
        assert_eq!(list_mode_names(true, "usage: x [--flag]\n"), None);
    }

    #[test]
    fn test_fns_need_the_attribute_within_three_lines() {
        let lines: Vec<String> = [
            "#[test]\n",
            "fn near() {}\n",
            "#[test]\n",
            "#[ignore]\n",
            "pub fn ignored() {}\n",
            "#[test]\n",
            "// one\n",
            "// two\n",
            "// three\n",
            "fn too_far() {}\n",
            "pub fn helper() {}\n",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        let want: HashSet<String> = ["near", "ignored"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert_eq!(test_fn_names(&lines), want);
    }

    #[test]
    fn section_lines_stop_at_the_next_heading() {
        let text = "## §5 x\nno\n## §6 Invariants\nin one\n\nin two\n## §7 next\nout\n";
        assert_eq!(section_lines(text), vec!["", "in one", "", "in two"]);
    }

    #[test]
    fn rs_surface_enumerates_the_test_fns_of_its_crate() {
        let tree = Tree::new(Path::new(FIXTURE_TREE));
        let want: HashSet<String> = ["alpha_ok", "ignored_case", "nested_case"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert_eq!(tree.names("rs:fixture-crate"), Some(want));
    }

    #[test]
    fn rs_surface_for_an_absent_crate_is_none() {
        let tree = Tree::new(Path::new(FIXTURE_TREE));
        assert_eq!(tree.names("rs:no-such-crate"), None);
    }

    // ---- C5: the §6 ids are PP-1..PP-max -------------------------------------------------

    #[test]
    fn c5_contiguous_ids_are_clean() {
        let rows = format!("{ARMED}\n{PP3}");
        let s = section6(&rows, &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 2);
        assert_eq!(verdict(&s.records), "CLEAN");
        assert!(check_id_contiguity(&spec6(&rows.replace("PP-3", "PP-2"))).is_empty());
        assert_eq!(verdict(&check_id_contiguity(&spec6(&rows))), "C5");
    }

    #[test]
    fn c5_names_the_first_gap_and_lists_every_gap() {
        let rows = "| PP-1 | r | m | h | RETIRED | x |\n| PP-2 | r | m | h | RETIRED | x |\n\
                    | PP-5 | r | m | h | RETIRED | x |";
        let got = check_id_contiguity(&spec6(rows));
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0].line(),
            "VIOLATION\tC5\tPP-3\t§6 is missing PP-3, PP-4 (ids run PP-1..PP-5). §0.6 keeps a \
             retired rule's number with status RETIRED, so a gap is an invariant that was DELETED \
             rather than retired"
        );
    }

    // ---- the live tree: NOT a unit test; run by hand against the Python original ----------

    #[test]
    #[ignore = "live parity: scans this checkout; set SEC6_PARITY_OUT to write the lines"]
    fn live_tree_records() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = root.join("docs/specifications");
        let mut specs: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("specifications dir")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with("PP-LLAMA-001-MASTER") && name.ends_with(".md")
            })
            .collect();
        specs.sort();
        assert_eq!(specs.len(), 1, "the master is the one spec this joins");
        let records = scan(&root, &specs[0]).expect("scan");
        let out = lines(&records).join("\n") + "\n";
        if let Ok(path) = std::env::var("SEC6_PARITY_OUT") {
            std::fs::write(path, out).expect("write parity output");
        }
    }
}
