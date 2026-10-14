//! #5291: the no-test debt's case tables, plus the corpus gate that runs in
//! `workspace-test` (pass/fail, no report-only row).

use super::*;
use crate::schema::parse_contract_str;
use std::path::PathBuf;

const ZERO_TESTS: &str = r#"
metadata:
  version: "1.0.0"
  description: "fixture with no tests"
  references: ["Paper (2024)"]
"#;

fn with_one_test() -> String {
    format!(
        "{ZERO_TESTS}falsification_tests:\n  - id: F-1\n    rule: \"r\"\n    prediction: \"p\"\n    if_fails: \"f\"\n    test_harness: \"cargo test -p x t\"\n"
    )
}

/// The first listed path: a real debt entry, whatever the list holds today.
fn a_listed_path() -> &'static str {
    debt_entries()
        .into_iter()
        .next()
        .expect("the list is not empty until 0.75")
}

#[test]
fn debt_cap_case_table() {
    let rows = [
        ("0.70.3", Some(803)),
        ("0.71.0", Some(803)),
        ("0.71.2", Some(803)),
        ("0.72.0", Some(602)),
        ("0.72.0-rc.1", Some(602)),
        ("0.73.4", Some(401)),
        ("0.74.0", Some(200)),
        ("0.75.0", Some(0)),
        ("0.80.1", Some(0)),
        ("1.0.0", Some(0)),
        ("", None),
        ("x.y", None),
        ("0", None),
    ];
    for (version, want) in rows {
        assert_eq!(debt_cap(version), want, "{version:?}");
    }
}

#[test]
fn rel_to_contracts_case_table() {
    let rows: [(&str, Option<&str>); 11] = [
        ("contracts/a-v1.yaml", Some("a-v1.yaml")),
        ("/x/y/contracts/sub/a-v1.yaml", Some("sub/a-v1.yaml")),
        ("/x/contracts/y/contracts/a.yaml", Some("a.yaml")),
        (
            "contracts/entrenar/kaizen/a.yaml",
            Some("entrenar/kaizen/a.yaml"),
        ),
        ("contracts/publish-manifests/a.yaml", None),
        ("contracts/quarantine/a.yaml", None),
        ("/tmp/fixture/a.yaml", None),
        ("contracts", None),
        // Files the walker never reads as contracts, by name or extension.
        ("contracts/ontology.yaml", None),
        ("contracts/x/binding.yaml", None),
        ("contracts/a-v1.yml", None),
    ];
    for (path, want) in rows {
        assert_eq!(rel_to_contracts(Path::new(path)).as_deref(), want, "{path}");
    }
}

#[test]
fn schema_026_case_table() {
    let zero = parse_contract_str(ZERO_TESTS).expect("parses");
    let one = parse_contract_str(&with_one_test()).expect("parses");
    let listed = format!("contracts/{}", a_listed_path());
    // (row, path, contract, fires)
    let rows: [(&str, &str, &Contract, bool); 7] = [
        (
            "new contract, no tests",
            "contracts/brand-new-v1.yaml",
            &zero,
            true,
        ),
        (
            "new nested contract, no tests",
            "/r/contracts/x/new-v1.yaml",
            &zero,
            true,
        ),
        ("listed contract, no tests", &listed, &zero, false),
        (
            "new contract with a test",
            "contracts/brand-new-v1.yaml",
            &one,
            false,
        ),
        (
            "new kaizen contract, no tests (pv walks kaizen/)",
            "contracts/trueno/kaizen/new-v1.yaml",
            &zero,
            true,
        ),
        (
            "outside contracts/",
            "/tmp/fixture/new-v1.yaml",
            &zero,
            false,
        ),
        (
            "walker-skipped dir",
            "contracts/pipelines/new-v1.yaml",
            &zero,
            false,
        ),
    ];
    for (row, path, contract, fires) in rows {
        let v = check_in_tree(Path::new(path), contract);
        assert_eq!(v.is_some(), fires, "{row}: {v:?}");
        if let Some(v) = v {
            assert_eq!(v.rule, "SCHEMA-026", "{row}");
            assert_eq!(v.severity, Severity::Error, "{row}");
        }
    }
}

/// SCHEMA-026 reaches `pv validate` through `validate_artifact`, not only the
/// helper: a zero-test file written at `contracts/<new>.yaml` of a census tree
/// is refused, and the same file in a `contracts/` with no census is not
/// (another project's corpus, a scratch fixture).
#[test]
fn validate_artifact_refuses_unlisted_zero_test_contract_in_census_tree_only() {
    for (census, fires) in [(true, true), (false, false)] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("contracts");
        std::fs::create_dir_all(dir.join("sub")).expect("mkdir");
        if census {
            std::fs::write(dir.join(CENSUS_MARKER), "{}\n").expect("write census");
        }
        for rel in ["brand-new-v1.yaml", "sub/nested-new-v1.yaml"] {
            let path = dir.join(rel);
            std::fs::write(&path, ZERO_TESTS).expect("write");
            let (_, violations) = crate::schema::validate_artifact(&path).expect("validates");
            let fired = violations
                .iter()
                .any(|v| v.rule == "SCHEMA-026" && v.severity == Severity::Error);
            assert_eq!(fired, fires, "census={census} {rel}: {violations:?}");
        }
    }
}

/// The scope is not vacuous on this repo: the real `contracts/` carries the
/// census marker, so an unlisted zero-test contract there is refused, at the
/// top level and nested.
#[test]
fn the_repo_contracts_dir_is_a_census_tree() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let zero = parse_contract_str(ZERO_TESTS).expect("parses");
    for rel in [
        "brand-new-never-written-v1.yaml",
        "never-written-dir/brand-new-v1.yaml",
    ] {
        let new = root.join(rel);
        assert!(
            in_census_tree(&new),
            "{} has no {CENSUS_MARKER}",
            root.display()
        );
        assert!(
            check(&new, &zero).is_some_and(|v| v.rule == "SCHEMA-026"),
            "{rel}"
        );
    }
    let listed = root.join(a_listed_path());
    assert!(check(&listed, &zero).is_none(), "{}", listed.display());
}

#[test]
fn measure_reports_unlisted_stale_and_unwalked() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("contracts");
    for sub in ["", "x", "x/kaizen", "quarantine"] {
        std::fs::create_dir_all(dir.join(sub)).expect("mkdir");
    }
    let write = |rel: &str, body: &str| std::fs::write(dir.join(rel), body).expect("write");
    write("new-zero-v1.yaml", ZERO_TESTS);
    write("x/tested-v1.yaml", &with_one_test());
    write("binding.yaml", "bindings: []\n");
    write("x/kaizen/k-v1.yaml", ZERO_TESTS);
    write("quarantine/q-v1.yaml", ZERO_TESTS);
    write("x/old-v1.yml", ZERO_TESTS);
    let r = measure(&dir);
    assert_eq!(r.no_falsification_tests, 2);
    assert_eq!(
        r.unlisted,
        vec![
            "new-zero-v1.yaml".to_string(),
            "x/kaizen/k-v1.yaml".to_string()
        ]
    );
    assert_eq!(
        r.stale.len(),
        debt_entries().len(),
        "nothing listed exists here"
    );
    assert!(r.unparsed.is_empty(), "{:?}", r.unparsed);
    assert_eq!(
        r.unwalked, 2,
        "binding.yaml + a .yml; kaizen/ is walked, quarantine/ excluded"
    );
}

/// THE GATE. Over the real corpus: no unlisted zero-test contract, no stale
/// line, the list within this release's cap, and the unwalked count within its
/// cap. Red here is the tag refused (`CARGO_PKG_VERSION` is the workspace
/// version the release bumps).
#[test]
fn corpus_no_test_debt_within_release_cap() {
    let Some(dir): Option<PathBuf> = crate::tree::workspace_path_or_skip_at(
        "corpus_no_test_debt_within_release_cap",
        Path::new(env!("CARGO_MANIFEST_DIR")),
        "contracts",
    ) else {
        return;
    };
    let r = measure(&dir);
    // A repeated line would be absorbed by the set and escape the cap.
    let lines = NO_TEST_DEBT
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .count();
    assert_eq!(
        lines,
        debt_entries().len(),
        "no_test_debt.txt repeats a line"
    );
    assert!(
        r.unparsed.is_empty(),
        "contracts pv cannot parse: {:?}",
        r.unparsed
    );
    assert!(
        r.unlisted.is_empty(),
        "SCHEMA-026: zero-test contracts not on the debt list (add tests): {:?}",
        r.unlisted
    );
    assert!(
        r.stale.is_empty(),
        "stale no_test_debt.txt lines (the contract has tests now, or is gone; delete the line): {:?}",
        r.stale
    );
    let version = env!("CARGO_PKG_VERSION");
    let cap = debt_cap(version).expect("workspace version parses");
    let n = debt_entries().len();
    assert_eq!(n, r.no_falsification_tests, "list and corpus agree");
    assert!(
        n <= cap,
        "no-test debt {n} exceeds the {version} cap {cap} (#5291 four-part split)"
    );
    assert!(
        r.unwalked <= UNWALKED_CAP,
        "unwalked .yaml under contracts/ rose to {} (cap {UNWALKED_CAP}); sort the new file on #5291",
        r.unwalked
    );
}
