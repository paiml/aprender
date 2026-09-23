//! ONT-2c (PMAT-4071, aprender#4071) — `pv ontology export --owl`, `pv ontology tbox`, `pv lint --gate tbox`,
//! driven through the BUILT binary. The lib tests prove the writer and the told-closure. This file proves the
//! CLI surface the row's probe calls, and the one mapping the row names at the dispatch layer: the gate maps
//! the advisory report to `Unknown{Advisory}` (`decline: Advisory`, exit 2) and has NO arm that exits 0 (R-7).
//!
//! DISCRIMINATION: a fresh corpus declines with exit 2, and a stale or missing artifact is exit 3. A build
//! that mapped Advisory to a pass (exit 0) fails `the_repo_corpus_is_advisory_never_zero`. One that stopped
//! checking freshness fails `a_stale_report_is_refused_with_exit_3`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let out = Command::new(pv_bin())
        .current_dir(scratch.path())
        .args(args)
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn show(r: &Run) -> String {
    format!(
        "exit {}\n--- stdout\n{}\n--- stderr\n{}",
        r.code, r.stdout, r.stderr
    )
}

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn s(p: &Path) -> String {
    p.to_str().expect("utf-8 path").to_string()
}

/// A corpus directory holding the fixture Σ plus the two artifacts `pv ontology … --write` produces.
fn fresh_fixture_corpus() -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        repo("tests/fixtures/ont/owl/ontology.yaml"),
        d.path().join("ontology.yaml"),
    )
    .expect("copy Σ");
    // `pv lint` declines a corpus with 0 contracts before any gate runs; one minimal contract makes it a corpus.
    std::fs::copy(
        repo("tests/fixtures/ont/sigma-ok/fixture-kernel-v1.yaml"),
        d.path().join("fixture-kernel-v1.yaml"),
    )
    .expect("copy contract");
    let sigma = s(&d.path().join("ontology.yaml"));
    let r = pv(&["ontology", "export", "--owl", "--write", &sigma]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let r = pv(&["ontology", "tbox", "--write", &sigma]);
    assert_eq!(r.code, 0, "{}", show(&r));
    d
}

#[test]
fn export_prints_exactly_the_committed_fixture_axiom_set() {
    let r = pv(&[
        "ontology",
        "export",
        "--owl",
        &s(&repo("tests/fixtures/ont/owl/ontology.yaml")),
    ]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let want =
        std::fs::read_to_string(repo("tests/fixtures/ont/owl/expected.ofn")).expect("expected.ofn");
    assert_eq!(r.stdout, want);
}

#[test]
fn export_without_a_format_is_refused() {
    let r = pv(&[
        "ontology",
        "export",
        &s(&repo("tests/fixtures/ont/owl/ontology.yaml")),
    ]);
    assert_eq!(r.code, 2, "{}", show(&r));
}

#[test]
fn the_repo_export_equals_the_tracked_ofn() {
    // The row's probe: `"$PV" ontology export --owl contracts/ontology.yaml | cmp - contracts/ontology.ofn`.
    let r = pv(&[
        "ontology",
        "export",
        "--owl",
        &s(&repo("contracts/ontology.yaml")),
    ]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let tracked =
        std::fs::read_to_string(repo("contracts/ontology.ofn")).expect("contracts/ontology.ofn");
    assert_eq!(
        r.stdout, tracked,
        "contracts/ontology.ofn is not what the writer produces"
    );
}

#[test]
fn write_writes_both_artifacts_next_to_sigma() {
    let d = fresh_fixture_corpus();
    let ofn =
        std::fs::read_to_string(d.path().join("ontology.ofn")).expect("--write wrote ontology.ofn");
    assert!(ofn.contains("SymmetricObjectProperty("), "{ofn}");
    let rep: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(d.path().join("tbox-report.json")).expect("report"),
    )
    .expect("report is JSON");
    assert_eq!(rep["advisory"], true);
    assert_eq!(rep["method"], "told-closure");
    assert_eq!(rep["consistent"], true);
}

#[test]
fn the_repo_corpus_is_advisory_never_zero() {
    let r = pv(&["lint", &s(&repo("contracts")), "--gate", "tbox"]);
    assert_eq!(
        r.code,
        2,
        "the tbox gate must DECLINE (Unknown{{Advisory}}), never pass: {}",
        show(&r)
    );
    assert!(
        r.stderr.contains("decline: Advisory") || r.stdout.contains("decline: Advisory"),
        "{}",
        show(&r)
    );
}

#[test]
fn a_fresh_fixture_corpus_is_advisory() {
    let d = fresh_fixture_corpus();
    let r = pv(&["lint", &s(d.path()), "--gate", "tbox"]);
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(
        format!("{}{}", r.stdout, r.stderr).contains("Advisory"),
        "{}",
        show(&r)
    );
}

#[test]
fn a_stale_report_is_refused_with_exit_3() {
    let d = fresh_fixture_corpus();
    let p = d.path().join("tbox-report.json");
    let body = std::fs::read_to_string(&p).expect("report");
    std::fs::write(&p, body.replace("\"classes\": 3", "\"classes\": 9")).expect("tamper");
    let r = pv(&["lint", &s(d.path()), "--gate", "tbox"]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.contains("tbox-report.json"), "{}", show(&r));
}

#[test]
fn a_missing_ofn_is_refused_with_exit_3() {
    let d = fresh_fixture_corpus();
    std::fs::remove_file(d.path().join("ontology.ofn")).expect("rm");
    let r = pv(&["lint", &s(d.path()), "--gate", "tbox"]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.contains("ontology.ofn"), "{}", show(&r));
}

#[test]
fn an_undeclared_unexpressed_key_is_exit_3() {
    let d = tempfile::tempdir().expect("tempdir");
    let sigma = std::fs::read_to_string(repo("tests/fixtures/ont/owl/ontology.yaml")).expect("Σ");
    let broken = sigma.replace("  - {key: symbols, reader: ontology/owl.rs}\n", "");
    assert_ne!(broken, sigma, "the mutation must change Σ");
    std::fs::write(d.path().join("ontology.yaml"), broken).expect("write");
    let r = pv(&[
        "ontology",
        "export",
        "--owl",
        &s(&d.path().join("ontology.yaml")),
    ]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.contains("symbols"), "{}", show(&r));
}
