//! The merge driver's case table (the old `roadmap_merge.py --selftest`): one `PASS  <row>` /
//! `FAIL  <row>` line per row. The sorted-order rows are judged by the repo's own
//! `scripts/check_roadmap_sorted.sh` over the merged file, and the mutation row demands that
//! checker exit EXACTLY 1 (a 2 or 127 satisfies "nonzero" vacuously).

use std::path::Path;
use std::process::Command;

use super::merge::merge;
use super::split_entries;

const H: &str = "roadmap_version: '1.0'\n# a header comment\nroadmap:\n";
const E1: &str = "- id: PMAT-1\n  a: 1\n";
const E2: &str = "- id: PMAT-2\n  a: 2\n";
const E3: &str = "- id: PMAT-3\n  a: 3\n";
const E4: &str = "- id: PMAT-4\n  a: 4\n";
const E5: &str = "- id: PMAT-5\n  a: 5\n";
const E10: &str = "- id: PMAT-10\n  a: 10\n";
const E1_EDIT: &str = "- id: PMAT-1\n  a: 11\n  note: edited by ours\n";

type Assertion = fn(&str, &str) -> Vec<String>;

/// One case: name, base, ours, theirs, expect-conflict, expect-checker-fail, assertion.
pub struct Row {
    pub name: &'static str,
    pub base: String,
    pub ours: String,
    pub theirs: String,
    pub expect_conflict: bool,
    pub expect_checker_fail: bool,
    pub assertion: Option<Assertion>,
}

fn ids_in_order(text: &str) -> Vec<String> {
    split_entries(text).1.into_iter().map(|e| e.0).collect()
}

fn a_both_appended(out: &str, _err: &str) -> Vec<String> {
    let ids = ids_in_order(out);
    if ids == ["PMAT-1", "PMAT-2", "PMAT-3"] {
        return vec![];
    }
    vec![format!(
        "both ids must be present and ascending, got {ids:?}"
    )]
}

fn a_edit_verbatim(out: &str, _err: &str) -> Vec<String> {
    if out.contains(E1_EDIT) {
        return vec![];
    }
    vec![format!(
        "ours' edited block was not taken byte-for-byte: {out:?}"
    )]
}

fn a_conflict_names_id(_out: &str, err: &str) -> Vec<String> {
    if err.contains("PMAT-1") {
        return vec![];
    }
    vec![format!("stderr must name the conflicting id, got {err:?}")]
}

fn a_conflict_names_id2(_out: &str, err: &str) -> Vec<String> {
    if err.contains("PMAT-2") {
        return vec![];
    }
    vec![format!("stderr must name the conflicting id, got {err:?}")]
}

fn a_entry_gone(out: &str, _err: &str) -> Vec<String> {
    if out.contains("PMAT-2") {
        return vec![format!(
            "the uncontested deletion did not remove PMAT-2: {out:?}"
        )];
    }
    let ids = ids_in_order(out);
    if ids == ["PMAT-1"] {
        return vec![];
    }
    vec![format!("surviving ids: {ids:?}")]
}

fn a_header_verbatim(out: &str, _err: &str) -> Vec<String> {
    let Some(first) = out.find("- id:") else {
        return vec!["no entry in the output at all".to_string()];
    };
    if &out[..first] == H {
        return vec![];
    }
    vec![format!("header changed: {:?} != {H:?}", &out[..first])]
}

fn a_tail_append_sorted(out: &str, _err: &str) -> Vec<String> {
    let ids = ids_in_order(out);
    if ids == ["PMAT-1", "PMAT-3", "PMAT-4", "PMAT-5"] {
        return vec![];
    }
    vec![format!(
        "out-of-order tail appends must be re-seated, got {ids:?}"
    )]
}

fn row(
    name: &'static str,
    abc: [String; 3],
    flags: (bool, bool),
    assertion: Option<Assertion>,
) -> Row {
    let [base, ours, theirs] = abc;
    Row {
        name,
        base,
        ours,
        theirs,
        expect_conflict: flags.0,
        expect_checker_fail: flags.1,
        assertion,
    }
}

fn cat(parts: &[&str]) -> String {
    parts.concat()
}

/// The eight rows.
#[must_use]
pub fn rows() -> Vec<Row> {
    let e1_12 = E1.replace("a: 1", "a: 12");
    let e2_22 = E2.replace("a: 2", "a: 22");
    vec![
        row(
            "both-append",
            [cat(&[H, E1]), cat(&[H, E1, E2]), cat(&[H, E1, E3])],
            (false, false),
            Some(a_both_appended),
        ),
        row(
            "edit",
            [cat(&[H, E1]), cat(&[H, E1_EDIT]), cat(&[H, E1])],
            (false, false),
            Some(a_edit_verbatim),
        ),
        row(
            "conflict",
            [cat(&[H, E1]), cat(&[H, E1_EDIT]), cat(&[H, &e1_12])],
            (true, false),
            Some(a_conflict_names_id),
        ),
        row(
            "del-unc",
            [cat(&[H, E1, E2]), cat(&[H, E1]), cat(&[H, E1, E2])],
            (false, false),
            Some(a_entry_gone),
        ),
        row(
            "del-edit",
            [cat(&[H, E1, E2]), cat(&[H, E1]), cat(&[H, E1, &e2_22])],
            (true, false),
            Some(a_conflict_names_id2),
        ),
        row(
            "header-untouched",
            [cat(&[H, E1]), cat(&[H, E1, E2]), cat(&[H, E1, E3])],
            (false, false),
            Some(a_header_verbatim),
        ),
        row(
            "tail-append-unsorted-input",
            [
                cat(&[H, E1, E5]),
                cat(&[H, E1, E5, E3]),
                cat(&[H, E1, E5, E4]),
            ],
            (false, false),
            Some(a_tail_append_sorted),
        ),
        row(
            "mut",
            [cat(&[H, E1]), cat(&[H, E1, E10]), cat(&[H, E1, E2])],
            (false, true),
            None,
        ),
    ]
}

fn checker_verdict(td: &Path, out_path: &Path, checker: &Path, expect_fail: bool) -> Vec<String> {
    let empty = td.join(".empty-template");
    let _ = std::fs::create_dir_all(&empty);
    let _ = Command::new("git")
        .arg("init")
        .arg("-q")
        .arg(format!("--template={}", empty.display()))
        .arg(td)
        .output();
    let run = Command::new("bash")
        .arg(checker)
        .arg(out_path)
        .current_dir(td)
        .output();
    let Ok(o) = run else {
        return vec![format!(
            "check_roadmap_sorted.sh could not run: {checker:?}"
        )];
    };
    let rc = o.status.code().unwrap_or(-1);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    if expect_fail {
        if rc != 1 {
            return vec![format!("check_roadmap_sorted.sh rc={rc}, expected EXACTLY 1 (a 2/127 would satisfy 'nonzero' vacuously): {text:?}")];
        }
        if !text.contains("FAIL") {
            return vec![format!("checker exited 1 without a FAIL line: {text:?}")];
        }
        return vec![];
    }
    if rc != 0 {
        return vec![format!(
            "check_roadmap_sorted.sh rc={rc} on the merged file: {text:?}"
        )];
    }
    vec![]
}

/// The failures of one row (empty = PASS).
#[must_use]
pub fn check_row(r: &Row, checker: &Path) -> Vec<String> {
    // The mutation row runs the driver with the numeral-as-string hook on.
    let result = merge(&r.base, &r.ours, &r.theirs, r.expect_checker_fail);
    let conflicted = result.is_err();
    if conflicted != r.expect_conflict {
        return vec![format!(
            "driver rc={} (expected {}), output: {:?}",
            i32::from(conflicted),
            if r.expect_conflict { "nonzero" } else { "0" },
            result
                .as_ref()
                .err()
                .map(|c| c.0.clone())
                .unwrap_or_default()
        )];
    }
    let (merged, err) = match &result {
        Ok(m) => (m.clone(), String::new()),
        Err(c) => (String::new(), format!("conflict: {}\n", c.0)),
    };
    let mut fails = r.assertion.map(|a| a(&merged, &err)).unwrap_or_default();
    if r.expect_conflict {
        return fails;
    }
    // std scratch dir, not the tempfile crate: this runs inside the shipped bin, and tempfile
    // is a dev-dependency of the published pv crate.
    let td = std::env::temp_dir().join(format!("pv-roadmap-selftest-{}", std::process::id()));
    if std::fs::create_dir_all(&td).is_err() {
        return vec!["no scratch directory".to_string()];
    }
    let out_path = td.join("roadmap.yaml");
    if std::fs::write(&out_path, &merged).is_err() {
        let _ = std::fs::remove_dir_all(&td);
        return vec!["the driver wrote no output file".to_string()];
    }
    fails.extend(checker_verdict(
        &td,
        &out_path,
        checker,
        r.expect_checker_fail,
    ));
    let _ = std::fs::remove_dir_all(&td);
    fails
}

/// Run the table, printing `PASS  <row>` / `FAIL  <row>` lines. Exit 0 all green, 1 any red,
/// 2 when the checker script is missing.
#[must_use]
pub fn run_selftest(checker: &Path) -> i32 {
    if !checker.is_file() {
        eprintln!(
            "ENV: {} is missing; the order rows cannot be decided",
            checker.display()
        );
        return 2;
    }
    let table = rows();
    let mut red = 0;
    for r in &table {
        let fails = check_row(r, checker);
        if fails.is_empty() {
            println!("PASS  {}", r.name);
            continue;
        }
        red += 1;
        println!("FAIL  {}", r.name);
        for line in fails.iter().flat_map(|f| f.lines()) {
            println!("        {line}");
        }
    }
    println!("{}/{} rows, {} failed", table.len() - red, table.len(), red);
    i32::from(red > 0)
}
