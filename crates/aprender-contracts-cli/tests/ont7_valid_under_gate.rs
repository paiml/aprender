//! ONT-7 (PMAT-4076) — `pv lint --gate valid-under`: kernel-kind contracts carry a world index.
//!
//! The row's probe, verbatim (paiml/infra `docs/specifications/paiml-ontology.md` v4.12 :657):
//! `jq -e '.contracts_without_valid_under!=null' contracts/lint-baseline.json && pv lint contracts/ --gate
//! valid-under --format json … jq -e '.verdict=="Pass"'`. The first test is that probe on the real corpus.
//!
//! Every rule has a fixture that fires it, because on the real corpus ONE contract carries `valid_under`
//! (`ont-verdict-lattice-v1`) and a rule that cannot fire there is only observable here:
//!
//! | fixture | expected |
//! |---|---|
//! | `valid-under-ok` | exit 0, Pass — world + every qualifier well-formed |
//! | `valid-under-unknown-world` | exit 1, PV-ONT-014 — `world: mars`, not in Σ |
//! | `valid-under-appendix-b` | exit 0, Pass — the spec's Appendix B example verbatim: no `world`, reads `committed` |
//! | `valid-under-world-not-a-string` | exit 1, PV-ONT-014 |
//! | `valid-under-undeclared-key` | exit 1, PV-ONT-013 — closed key set |
//! | `valid-under-empty` | exit 1, PV-ONT-013 — `valid_under: {}` |
//! | `valid-under-nonkernel-bad` | exit 1, PV-ONT-014 — no kernel at all, and still a reject, not a decline |
//! | `valid-under-bad-qualifier` | exit 1, PV-ONT-015 — `backend: []`, non-string toolchain version |
//! | `valid-under-ratchet-rise` | exit 1, PV-ONT-016 — one kernel without, baseline 0 |
//! | `valid-under-no-kernels` | exit 2, decline — nothing the row obliges |
//! | `sigma-absent` | exit 2, decline — no world index to resolve into |
//!
//! DISCRIMINATION: `valid-under-ok` passes at exit 0 and every reject names its own rule id, so a build that
//! refuses everything, or refuses with the wrong rule, fails this file.

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

fn fixture(name: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(name);
    p.to_str().expect("utf-8 path").to_string()
}

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn gate(dir: &str) -> Run {
    pv(&["lint", dir, "--gate", "valid-under", "--format", "json"])
}

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

/// The rejects: exit 1, verdict Fail, and the report names exactly this rule.
fn assert_rejects_with(fixture_name: &str, rule: &str) {
    let r = gate(&fixture(fixture_name));
    assert_eq!(r.code, 1, "{fixture_name} must reject\n{}", show(&r));
    assert_eq!(json_of(&r)["verdict"], "Fail", "{}", show(&r));
    assert!(
        r.stdout.contains(rule),
        "{fixture_name} must reject with {rule}\n{}",
        show(&r)
    );
    for other in ["PV-ONT-013", "PV-ONT-014", "PV-ONT-015", "PV-ONT-016"] {
        if other != rule {
            assert!(
                !r.stdout.contains(other),
                "{fixture_name} fired {other} as well as {rule} — one fixture, one rule\n{}",
                show(&r)
            );
        }
    }
}

#[test]
fn the_row_probe_passes_on_the_repo_corpus() {
    let baseline: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo_contracts().join("lint-baseline.json")).expect("baseline"),
    )
    .expect("baseline is JSON");
    assert!(
        baseline["contracts_without_valid_under"].is_u64(),
        "the probe reads a TOP-LEVEL contracts_without_valid_under"
    );
    let r = gate(repo_contracts().to_str().expect("utf-8 path"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["gate"], "valid-under", "{}", show(&r));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    let extra = &v["extra"];
    // Not vacuous: the corpus has kernel contracts, and at least one real one resolves its world.
    assert!(extra["kernel_contracts"].as_u64() > Some(0), "{}", show(&r));
    assert!(
        extra["by_world"]
            .as_array()
            .expect("by_world")
            .iter()
            .any(|w| w.as_str().is_some_and(|w| w.starts_with("committed="))),
        "ont-verdict-lattice-v1 carries world: committed\n{}",
        show(&r)
    );
    // The ratchet's own rule, not equality: a PR that annotates contracts lowers the count and passes whether
    // or not it also lowers the baseline (as `formal_prose` does), so this never forces a hand-edit.
    assert!(
        extra["contracts_without_valid_under"].as_u64() <= extra["baseline"].as_u64(),
        "the debt may not rise above the recorded baseline\n{}",
        show(&r)
    );
}

#[test]
fn a_well_formed_world_index_passes() {
    let r = gate(&fixture("valid-under-ok"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["extra"]["by_world"][0], "committed=1", "{}", show(&r));
}

#[test]
fn a_world_sigma_does_not_declare_rejects() {
    assert_rejects_with("valid-under-unknown-world", "PV-ONT-014");
}

#[test]
fn the_specs_own_appendix_b_example_passes_and_reads_committed() {
    let r = gate(&fixture("valid-under-appendix-b"));
    assert_eq!(r.code, 0, "{}", show(&r));
    assert_eq!(
        json_of(&r)["extra"]["by_world"][0],
        "committed=1",
        "{}",
        show(&r)
    );
}

#[test]
fn a_world_that_is_not_a_string_rejects() {
    assert_rejects_with("valid-under-world-not-a-string", "PV-ONT-014");
}

#[test]
fn an_empty_valid_under_rejects() {
    assert_rejects_with("valid-under-empty", "PV-ONT-013");
}

#[test]
fn a_bad_valid_under_rejects_even_with_no_kernel_contract() {
    assert_rejects_with("valid-under-nonkernel-bad", "PV-ONT-014");
}

/// R-8: a new ONT gate is COMPUTED in every run and armed per repo, like sigma, relations and shapes.
#[test]
fn the_full_lint_run_computes_valid_under() {
    let r = pv(&[
        "lint",
        repo_contracts().to_str().expect("utf-8 path"),
        "--format",
        "json",
    ]);
    let v = json_of(&r);
    let gate = v["gates"]
        .as_array()
        .expect("gates")
        .iter()
        .find(|g| g["name"] == "valid-under")
        .unwrap_or_else(|| panic!("the full run did not compute valid-under\n{}", show(&r)));
    assert_eq!(gate["verdict"], "Pass", "{}", show(&r));
}

#[test]
fn an_undeclared_valid_under_key_rejects() {
    assert_rejects_with("valid-under-undeclared-key", "PV-ONT-013");
}

#[test]
fn a_malformed_qualifier_rejects() {
    assert_rejects_with("valid-under-bad-qualifier", "PV-ONT-015");
    let r = gate(&fixture("valid-under-bad-qualifier"));
    assert!(
        r.stdout.matches("PV-ONT-015").count() >= 2,
        "both the empty backend list and the non-string toolchain version are named\n{}",
        show(&r)
    );
}

#[test]
fn a_rise_in_the_debt_rejects() {
    assert_rejects_with("valid-under-ratchet-rise", "PV-ONT-016");
}

#[test]
fn a_corpus_with_no_kernel_contract_declines() {
    let r = gate(&fixture("valid-under-no-kernels"));
    assert_eq!(
        r.code,
        2,
        "zero is a decline, never an accept\n{}",
        show(&r)
    );
    assert!(r.stderr.contains("no kernel-kind contract"), "{}", show(&r));
}

#[test]
fn a_corpus_with_no_sigma_declines() {
    let r = gate(&fixture("sigma-absent"));
    assert_eq!(r.code, 2, "{}", show(&r));
}
