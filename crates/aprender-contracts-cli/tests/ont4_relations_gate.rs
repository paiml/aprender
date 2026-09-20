//! ONT-4 (PMAT-3487) — `pv lint --gate relations`: typed relations, reporting into ONT-6's lattice.
//!
//! ONT-001 v4.5 §5 ONT-4 RED, verbatim: "four relations accepted; `contradicts` symmetric; cycle in
//! `supersedes|depends_on` → reject naming it; dangling id → reject; domain/range per Σ → reject." Plus the exit
//! vocabulary every named gate shares: Pass 0 · reject 1 · decline 2 (no Σ, or no typed relation — R-2) · error 3
//! (malformed Σ).
//!
//! **`supersedes` and `contradicts` have no true instance in the real corpus**, so `relations-ok/` is the only
//! witness that the gate accepts them; the real corpus carries `depends_on` on the ONT contracts themselves,
//! which is why `the_repo_corpus_passes_the_relations_gate` has focus nodes at all.
//!
//! DISCRIMINATION: `relations-ok/` must PASS at exit 0, so a build that rejects every corpus fails this file.

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

fn gate(dir: &Path) -> Run {
    pv(&["lint", &s(dir), "--gate", "relations", "--format", "json"])
}

#[test]
fn the_repo_corpus_passes_the_relations_gate() {
    let r = gate(&repo_contracts());
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["gate"], "relations", "{}", show(&r));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    // R-2 by construction: the real corpus must carry typed relations, or this would be a decline
    assert!(
        v["extra"]["relations_n"].as_u64().unwrap_or(0) > 0,
        "the real corpus has typed relations\n{}",
        show(&r)
    );
    // the legacy field is measured, never rewritten: 188 contracts carry it at 3409b29d
    assert!(
        v["extra"]["legacy_depends_on"].as_u64().unwrap_or(0) > 100,
        "metadata.depends_on is counted\n{}",
        show(&r)
    );
}

#[test]
fn four_relations_are_accepted_and_contradicts_is_read_both_ways() {
    let r = gate(&fixture("relations-ok"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    // a refines b · b supersedes c · a depends_on c · a contradicts d (+ d contradicts a, materialized) = 5
    assert_eq!(v["extra"]["relations_n"], 5, "{}", show(&r));
    let roles = v["extra"]["roles_used"].to_string();
    for want in ["refines=1", "supersedes=1", "depends_on=1", "contradicts=2"] {
        assert!(roles.contains(want), "{want} in {roles}\n{}", show(&r));
    }
}

#[test]
fn a_dangling_id_rejects_at_exit_1_naming_it() {
    let r = gate(&fixture("relations-dangling"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert_eq!(json_of(&r)["verdict"], "Fail");
    assert!(
        r.stdout.contains("`a` depends_on `ghost`"),
        "contract, role and target are named\n{}",
        show(&r)
    );
}

#[test]
fn a_cycle_through_an_acyclic_role_rejects_at_exit_1_naming_the_path() {
    let r = gate(&fixture("relations-cycle"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stdout.contains("a -> b -> c -> a"),
        "the closing path is named\n{}",
        show(&r)
    );
    assert!(r.stdout.contains("PV-ONT-009"), "{}", show(&r));
}

#[test]
fn a_role_that_is_not_contract_to_contract_rejects_at_exit_1() {
    let r = gate(&fixture("relations-domain"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-ONT-006"), "{}", show(&r));
    assert!(r.stdout.contains("`binds`"), "{}", show(&r));
}

#[test]
fn an_undeclared_role_and_a_non_list_reject_at_exit_1() {
    let r = gate(&fixture("relations-malformed"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-ONT-005"), "{}", show(&r));
    assert!(r.stdout.contains("PV-ONT-007"), "{}", show(&r));
}

/// R-2: a corpus with Σ but not one typed relation measured nothing — decline, never Pass.
#[test]
fn a_corpus_with_no_typed_relations_declines_at_exit_2() {
    let r = gate(&fixture("sigma-ok"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: "), "{}", show(&r));
}

#[test]
fn a_corpus_without_sigma_declines_at_exit_2() {
    let r = gate(&fixture("sigma-absent"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: "), "{}", show(&r));
}

#[test]
fn a_malformed_sigma_is_an_error_at_exit_3() {
    let r = gate(&fixture("sigma-malformed"));
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.starts_with("error: "), "{}", show(&r));
}

/// The legacy ratchet: a rise above the recorded baseline is PV-ONT-010; the count itself never rejects.
#[test]
fn the_legacy_ratchet_rejects_a_rise_and_passes_a_hold() {
    let hold = gate(&fixture("relations-legacy"));
    assert_eq!(hold.code, 0, "{}", show(&hold));
    assert_eq!(json_of(&hold)["extra"]["legacy_unresolved_depends_on"], 1);
    let rise = gate(&fixture("relations-legacy-rise"));
    assert_eq!(rise.code, 1, "{}", show(&rise));
    assert!(rise.stdout.contains("PV-ONT-010"), "{}", show(&rise));
}

/// R-8: the gate is computed in every `pv lint` run, not only under `--gate`.
#[test]
fn the_full_run_computes_the_relations_gate() {
    let dir = s(&repo_contracts());
    let r = pv(&["lint", &dir, "--format", "json"]);
    let v = json_of(&r);
    let names: Vec<String> = v["gates"]
        .as_array()
        .map(|g| {
            g.iter()
                .filter_map(|x| x["name"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        names.iter().any(|n| n == "relations"),
        "{names:?}\n{}",
        show(&r)
    );
}
