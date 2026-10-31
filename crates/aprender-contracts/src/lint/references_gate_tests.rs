//! aprender#6397 — `references-resolve`, each case on a throwaway repo.

use super::*;
use crate::lint::GateResult;

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
    std::fs::write(p, text).expect("write");
}

fn contract(refs: &[&str]) -> String {
    let list: Vec<String> = refs.iter().map(|r| format!("    - \"{r}\"")).collect();
    format!(
        "metadata:\n  version: \"1.0.0\"\n  kind: kernel\n  description: fixture\n  references:\n{}\n\
         equations:\n  identity:\n    formula: \"len(out) = len(x)\"\n",
        list.join("\n")
    )
}

fn run(root: &Path) -> (GateResult, Vec<String>) {
    match run_references_gate(&root.join("contracts")) {
        RatchetOutcome::Ran { result, findings } => {
            (*result, findings.into_iter().map(|f| f.rule_id).collect())
        }
        RatchetOutcome::Declined(why) => panic!("expected a verdict, got a decline: {why}"),
    }
}

fn dead_of(r: &GateResult) -> Vec<String> {
    match r.extra.as_ref() {
        Some(GateExtra::ReferencesResolve { dead, .. }) => dead.clone(),
        other => panic!("expected ReferencesResolve, got {other:?}"),
    }
}

fn repo(refs: &[&str]) -> tempfile::TempDir {
    let t = tempfile::tempdir().expect("tempdir");
    write(t.path(), "crates/x/src/lib.rs", "");
    write(t.path(), "contracts/a-v1.yaml", &contract(refs));
    t
}

#[test]
fn classifier_case_table() {
    let t = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(t.path().join("crates")).expect("mkdir");
    let r = t.path();
    for (entry, want) in [
        ("crates/x/src/lib.rs", Some("crates/x/src/lib.rs")),
        ("crates/x/src/lib.rs::foo", Some("crates/x/src/lib.rs")),
        (
            "crates/x/src/lib.rs:42 (the fn)",
            Some("crates/x/src/lib.rs"),
        ),
        ("crates/x/src/lib.rs,", Some("crates/x/src/lib.rs")),
        ("crates/x/src/lib.rs.", Some("crates/x/src/lib.rs")),
        ("crates/x/src/gone.rs).", Some("crates/x/src/gone.rs")),
        ("https://arxiv.org/abs/1706.03762", None),
        ("Vaswani et al. (2017) Attention Is All You Need", None),
        ("../trueno/src/a.rs", None),
        ("crates/x/{a,b}.rs", None),
        ("crates/*.rs", None),
        ("nodir/a.rs", None),
        ("crates/x/src", None),
    ] {
        assert_eq!(repo_path_of(entry, r).as_deref(), want, "{entry}");
    }
}

#[test]
fn dead_path_is_reported_and_live_one_is_not() {
    let t = repo(&[
        "crates/x/src/lib.rs::f",
        "crates/x/src/gone.rs",
        "Smith (2020) A Paper",
    ]);
    write(
        t.path(),
        "contracts/lint-baseline.json",
        r#"{"dead_references": 0}"#,
    );
    let (r, rules) = run(t.path());
    assert_eq!(dead_of(&r), vec!["a-v1: crates/x/src/gone.rs".to_string()]);
    assert_eq!(rules, vec!["PV-RAT-004".to_string()]);
    assert!(!r.passed);
}

#[test]
fn all_live_passes() {
    let t = repo(&["crates/x/src/lib.rs"]);
    write(
        t.path(),
        "contracts/lint-baseline.json",
        r#"{"dead_references": 0}"#,
    );
    let (r, rules) = run(t.path());
    assert!(r.passed && rules.is_empty());
    assert!(dead_of(&r).is_empty());
}

#[test]
fn baseline_tolerates_existing_debt_but_not_a_new_dead_path() {
    let t = repo(&["crates/x/src/gone.rs"]);
    write(
        t.path(),
        "contracts/lint-baseline.json",
        r#"{"dead_references": 1}"#,
    );
    assert!(run(t.path()).0.passed);
    write(
        t.path(),
        "contracts/b-v1.yaml",
        &contract(&["crates/x/src/gone2.rs"]),
    );
    let (r, rules) = run(t.path());
    assert!(!r.passed);
    assert_eq!(rules, vec!["PV-RAT-004".to_string()]);
}

#[test]
fn no_baseline_reports_but_does_not_judge() {
    let t = repo(&["crates/x/src/gone.rs"]);
    let (r, rules) = run(t.path());
    assert!(r.skipped && rules.is_empty());
    assert_eq!(dead_of(&r).len(), 1);
}

#[test]
fn no_path_reference_declines() {
    let t = repo(&["Smith (2020) A Paper"]);
    assert!(matches!(
        run_references_gate(&t.path().join("contracts")),
        RatchetOutcome::Declined(_)
    ));
}

#[test]
fn swapping_one_dead_path_for_another_fails_though_the_count_is_equal() {
    let t = repo(&["crates/x/src/gone2.rs"]);
    write(
        t.path(),
        "contracts/lint-baseline.json",
        r#"{"dead_references": 1, "dead_reference_set": ["a-v1: crates/x/src/gone.rs"]}"#,
    );
    let (r, rules) = run(t.path());
    assert!(!r.passed);
    assert_eq!(rules, vec!["PV-RAT-004".to_string()]);
    // The same debt, unswapped, passes.
    write(
        t.path(),
        "contracts/a-v1.yaml",
        &contract(&["crates/x/src/gone.rs"]),
    );
    assert!(run(t.path()).0.passed);
}

#[test]
fn a_trailing_period_is_trimmed_so_the_path_is_checked_not_skipped() {
    let t = repo(&["crates/x/src/gone.rs."]);
    write(
        t.path(),
        "contracts/lint-baseline.json",
        r#"{"dead_references": 0}"#,
    );
    assert_eq!(
        dead_of(&run(t.path()).0),
        vec!["a-v1: crates/x/src/gone.rs".to_string()]
    );
}
