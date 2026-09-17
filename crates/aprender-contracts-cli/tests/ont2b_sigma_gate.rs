//! ONT-2b (PMAT-3471) — `pv lint --gate sigma`: one named gate, reporting into ONT-6's lattice.
//!
//! ONT-001 v4.4 §5 ONT-2b RED, verbatim: "undeclared role → exit 1; undeclared symbol in `formal:` without `prose`
//! → exit 1; `entity.type` not in Σ `entity_types` → exit 1; an `entity_types` entry naming no extractor → exit 3;
//! Σ key with no reader → exit 3; `not_expressible` without `reader` → exit 3; `extractors[]` entry without
//! `reader` → exit 3." The symbol rule is P3; this file covers the rest plus the `--gate` mode itself.
//!
//! **The two rules that cannot fire on the real corpus each have a fixture here** (plan v2 ruling 4): today 0 of
//! 1792 contracts carry `entity:` and 0 carry `relations:`, so `sigma-entity-type-unknown/` and
//! `sigma-undeclared-role/` are the only places those rules are observable at all.
//!
//! DISCRIMINATION: `sigma-ok/` must PASS at exit 0, so a build that errors for every corpus fails this file.

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

/// `tests/fixtures/ont/<name>` — one corpus per rule.
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(name)
}

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn s(p: &Path) -> String {
    p.to_str().expect("utf-8 path").to_string()
}

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

#[test]
fn the_repo_corpus_passes_the_sigma_gate() {
    let dir = s(&repo_contracts());
    let r = pv(&["lint", &dir, "--gate", "sigma", "--format", "json"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["gate"], "sigma", "{}", show(&r));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
}

#[test]
fn a_well_formed_fixture_corpus_passes() {
    let dir = s(&fixture("sigma-ok"));
    let r = pv(&["lint", &dir, "--gate", "sigma", "--format", "json"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["gate"], "sigma");
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
}

/// Vacuous on the real corpus (0 of 1792 contracts carry `entity:`), so the fixture is the only witness.
#[test]
fn an_entity_type_not_in_sigma_rejects_at_exit_1() {
    let dir = s(&fixture("sigma-entity-type-unknown"));
    let r = pv(&["lint", &dir, "--gate", "sigma", "--format", "json"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Fail", "{}", show(&r));
    assert!(
        r.stdout.contains("ghost-entity"),
        "the report names the offending type\n{}",
        show(&r)
    );
}

/// Vacuous on the real corpus (0 of 1792 contracts carry `relations:`), so the fixture is the only witness.
#[test]
fn an_undeclared_role_rejects_at_exit_1() {
    let dir = s(&fixture("sigma-undeclared-role"));
    let r = pv(&["lint", &dir, "--gate", "sigma", "--format", "json"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Fail", "{}", show(&r));
    assert!(
        r.stdout.contains("ghost_role"),
        "the report names the offending role\n{}",
        show(&r)
    );
}

/// A malformed Σ is the DECLARATION's fault, not the corpus's: exit 3 `error:`, never `reject:`.
#[test]
fn a_malformed_sigma_is_an_error_at_exit_3() {
    let dir = s(&fixture("sigma-malformed"));
    let r = pv(&["lint", &dir, "--gate", "sigma", "--format", "json"]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(
        r.stderr.starts_with("error: "),
        "a malformed Σ is `error:`, not `reject:`\n{}",
        show(&r)
    );
    assert!(
        r.stderr.contains("pv_contract") && r.stderr.contains("reader"),
        "the message names the extractor and what it lacks\n{}",
        show(&r)
    );
}

/// No Σ, nothing measured: R-2 says zero is a decline, never an accept.
#[test]
fn a_corpus_without_sigma_declines_at_exit_2() {
    let dir = s(&fixture("sigma-absent"));
    let r = pv(&["lint", &dir, "--gate", "sigma", "--format", "json"]);
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(
        r.stderr.contains("decline: "),
        "a decline carries its reason\n{}",
        show(&r)
    );
}

#[test]
fn an_unknown_gate_name_is_refused() {
    let dir = s(&repo_contracts());
    let r = pv(&["lint", &dir, "--gate", "no-such-gate", "--format", "json"]);
    assert_ne!(r.code, 0, "{}", show(&r));
    assert!(
        r.stderr.contains("no-such-gate"),
        "the refusal names the gate asked for\n{}",
        show(&r)
    );
}

/// `--gate` selects ONE gate: the report is that gate's, not the whole run's.
#[test]
fn the_gate_mode_reports_only_the_named_gate() {
    let dir = s(&repo_contracts());
    let r = pv(&["lint", &dir, "--gate", "validate", "--format", "json"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["gate"], "validate", "{}", show(&r));
    assert!(v.get("verdict").is_some(), "{}", show(&r));
    assert!(
        v.get("gates").is_none(),
        "a single-gate report is not a whole LintReport\n{}",
        show(&r)
    );
}
