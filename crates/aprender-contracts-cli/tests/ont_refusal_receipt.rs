//! #3605 / PMAT-3605 — `removed_by` under contract, on the CLI.
//!
//! **Both arms, and the second one is the point.** A validator that only ever sees valid input is
//! indistinguishable from `exit 0`, so the table plants a **plausible but undeclared** sentinel
//! (`tbd`) and requires RED. Accepting `v0.70` proves nothing on its own.
//!
//! | fixture | `removed_by` | expected |
//! |---|---|---|
//! | `refusal-ok` | `v0.70`, `never`, `unscheduled` | **Pass** — the declared set, all three forms |
//! | `refusal-undeclared-sentinel` | `tbd` | Fail — the arm that matters |
//! | `refusal-sha` | `ddb5a15eb` | Fail — precise about the tree, silent about the boundary |
//! | `refusal-bare-version` | `0.70` | Fail — a shape that permits two spellings gets both |
//! | `refusal-patch-version` | `v0.70.1` | Fail — a patch is not a release boundary |
//! | `refusal-missing` | *(absent)* | Fail — every refusal answers for its own lifetime |
//! | `refusal-no-exit-code` | — | Fail — a refusal a caller cannot branch on |
//! | `refusal-terse-reason` | — | Fail — `reason: "broken"` names no defect |
//!
//! **Why `minCount 1` is safe here and manufactures nothing.** A required field with no escape
//! produces fabricated data — `pv validate` requires `kani_harnesses`, so authors invent one, and
//! apex#57 had `lean_theorem: Theorems.RowMerged` copied verbatim into twelve contracts, resolving
//! to nothing, gate green throughout. The escape is the closed sentinel set: `never` and
//! `unscheduled` let "there is legitimately nothing here" be SAID and still CHECKED. That is why
//! `refusal-ok` carries all three forms and `refusal-undeclared-sentinel` must be red — the escape
//! has to be open enough to be honest and closed enough to be a gate.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures/ont").join(name)
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    fn all(&self) -> String {
        format!(
            "exit {}\n--- stdout\n{}\n--- stderr\n{}",
            self.code, self.stdout, self.stderr
        )
    }
}

fn shapes_on(name: &str) -> Run {
    let contracts = fixture(name).join("contracts");
    let out = Command::new(pv_bin())
        .args([
            "lint",
            contracts.to_str().expect("utf-8 path"),
            "--gate",
            "shapes",
            "--format",
            "json",
        ])
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Every fixture whose `removed_by` is outside the declared set must FAIL, and the report must name
/// the focus node — a red that does not say which entry is a red nobody can act on.
fn assert_refused(name: &str) {
    let r = shapes_on(name);
    assert_eq!(r.code, 1, "{name} was not refused\n{}", r.all());
    assert!(
        r.stdout.contains("refusal-receipt-v1"),
        "{name}: the report must name the shape that fired\n{}",
        r.all()
    );
}

#[test]
fn all_three_declared_forms_conform() {
    // A release, and BOTH sentinels. If the escape did not work, a refusal with nothing to promise
    // would have to invent a release — which is the defect this contract exists to prevent.
    let r = shapes_on("refusal-ok");
    assert_eq!(r.code, 0, "{}", r.all());
    let v: serde_json::Value = serde_json::from_str(&r.stdout).expect("json report");
    assert_eq!(v["extra"]["violations"].as_u64(), Some(0), "{}", r.all());
    assert!(
        v["extra"]["by_shape"]
            .as_array()
            .expect("by_shape")
            .iter()
            .any(|s| s.as_str() == Some("refusal-receipt-v1=3")),
        "all three entries must be focus nodes\n{}",
        r.all()
    );
}

#[test]
fn a_plausible_but_undeclared_sentinel_is_refused() {
    // THE ARM THAT MATTERS. `tbd` is exactly what someone writes when the field is required and
    // they have nothing to say; if it passed, the closed set would be decoration.
    assert_refused("refusal-undeclared-sentinel");
}

#[test]
fn a_sha_is_refused_because_it_answers_a_different_question() {
    assert_refused("refusal-sha");
}

#[test]
fn a_shape_that_permits_two_spellings_would_get_both() {
    assert_refused("refusal-bare-version");
    assert_refused("refusal-patch-version");
}

#[test]
fn a_refusal_that_says_nothing_about_its_lifetime_is_refused() {
    assert_refused("refusal-missing");
}

#[test]
fn a_refusal_a_caller_cannot_branch_on_or_read_is_refused() {
    assert_refused("refusal-no-exit-code");
    assert_refused("refusal-terse-reason");
}

#[test]
fn accept_and_refuse_are_distinct_answers() {
    // A build that collapses them passes an all-invalid table and an all-valid one alike.
    assert_eq!(shapes_on("refusal-ok").code, 0);
    assert_eq!(shapes_on("refusal-undeclared-sentinel").code, 1);
}

#[test]
fn every_fixture_carries_the_real_contract_byte_for_byte() {
    // Without this, the shape could be widened in `contracts/` while the case table stayed green —
    // the mutation control's blind spot.
    let real = std::fs::read(repo_root().join("contracts/refusal-receipt-v1.yaml"))
        .expect("the real contract is in the tree");
    for name in [
        "refusal-ok",
        "refusal-undeclared-sentinel",
        "refusal-sha",
        "refusal-bare-version",
        "refusal-patch-version",
        "refusal-missing",
        "refusal-no-exit-code",
        "refusal-terse-reason",
    ] {
        let copy = std::fs::read(fixture(name).join("contracts/refusal-receipt-v1.yaml"))
            .unwrap_or_else(|e| panic!("{name} carries the contract: {e}"));
        assert_eq!(
            copy, real,
            "{name}'s copy of refusal-receipt-v1.yaml has drifted from contracts/"
        );
    }
}

#[test]
fn every_refusal_in_the_tree_names_a_verb_the_registry_carries() {
    // The join the contract declares: the registry is the universe, and a refusal naming a verb
    // `apr --help` does not expose is a dangling reference. Checked here rather than in the shape
    // because the implemented subset cannot reach across documents.
    let ledger: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join("evidence/verbs/refusals.json"))
            .expect("the ledger is in the tree"),
    )
    .expect("the ledger is JSON");
    let registry = std::fs::read_to_string(repo_root().join("contracts/apr-cli-commands-v1.yaml"))
        .expect("the registry is in the tree");
    let refusals = ledger["refusals"].as_array().expect("refusals[]");
    assert!(!refusals.is_empty(), "an empty ledger passes vacuously");
    for r in refusals {
        let verb = r["verb"].as_str().expect("verb is a string");
        assert!(
            registry.contains(&format!("- name: {verb}\n")),
            "refusal names verb {verb:?}, which contracts/apr-cli-commands-v1.yaml does not carry"
        );
    }
}
