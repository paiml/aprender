//! ONT-4c3 (PMAT-3577) — logit-parity receipts as focus nodes, on the CLI.
//!
//! The three answers `pv lint --gate shapes` may give about this family, each on its own fixture, so that a
//! build which collapses any two of them fails here:
//!
//! | fixture | exit | why |
//! |---|---|---|
//! | `parity-green` | 0 | a complete v2 receipt: one focus node, no violation |
//! | `parity-nocomparator` | 1 | the state all seven records were in before #3577 back-filled them |
//! | `parity-unknownkind` | 1 | a comparator kind the shape does not accept — this is what the `sh:in` MUTATION breaks |
//! | `parity-unmigrated` | 2 | a legacy record refused BY NAME: `Unknown{WrongCorpus}`, never Pass |
//! | `parity-denominator-drift` | 2 | 2 receipts, denominator 1 — a receipt added without bumping the count |
//!
//! DISCRIMINATION, in both directions. `parity-green` must PASS, so a build that declines every parity corpus
//! fails this file; `parity-nocomparator` and `parity-unknownkind` must FAIL, so a build whose shapes accept
//! anything fails it too. A table of only-invalid cases is passed by a validator that rejects everything.
//!
//! THE MUTATION. Widening `in: [llama_cpp, transformers, self]` to accept `oracle` makes
//! `parity-unknownkind` pass, and this file goes RED. The fixture copies of the contract are asserted
//! byte-identical to `contracts/parity-receipt-v2.yaml`, so mutating the real contract without the fixtures is
//! caught by the drift test instead — either way, RED. A mutation that no committed case can see is not a
//! control.

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

fn violations(r: &Run) -> usize {
    let v: serde_json::Value = serde_json::from_str(&r.stdout).expect("json report");
    v["extra"]["violations"].as_u64().unwrap_or(0) as usize
}

fn focus_nodes(r: &Run) -> usize {
    let v: serde_json::Value = serde_json::from_str(&r.stdout).expect("json report");
    v["extra"]["focus_nodes_n"].as_u64().unwrap_or(0) as usize
}

#[test]
fn a_complete_receipt_passes_with_one_focus_node() {
    let r = shapes_on("parity-green");
    assert_eq!(r.code, 0, "{}", r.all());
    assert_eq!(violations(&r), 0, "{}", r.all());
    assert_eq!(focus_nodes(&r), 1, "{}", r.all());
    // The shapes are ARMED in the fixture (no lint-baseline.json narrows them), so a violation here would be
    // a real Fail rather than a reported one — which is what makes the two FAIL cases below meaningful.
    assert!(r.stdout.contains("parity-receipt-complete"), "{}", r.all());
}

#[test]
fn a_receipt_with_no_comparator_fails_naming_the_property() {
    // The state every one of the seven records was in before #3577: a self-comparison implying an oracle.
    let r = shapes_on("parity-nocomparator");
    assert_eq!(r.code, 1, "{}", r.all());
    assert_eq!(violations(&r), 1, "exactly one violation\n{}", r.all());
    assert!(r.stdout.contains("parity/comparator"), "{}", r.all());
    assert!(r.stdout.contains("minCount"), "{}", r.all());
}

#[test]
fn a_comparator_kind_the_shape_does_not_accept_fails() {
    // THE MUTATION TARGET. Widen `in:` to accept `oracle` and this case goes green — so this assertion is
    // what makes that mutation detectable.
    let r = shapes_on("parity-unknownkind");
    assert_eq!(r.code, 1, "{}", r.all());
    assert!(violations(&r) >= 1, "{}", r.all());
    assert!(
        r.stdout.contains("llama_cpp") || r.stdout.contains("\"in\"") || r.stdout.contains("(in)"),
        "the report must name the `in` constraint that rejected the kind\n{}",
        r.all()
    );
}

#[test]
fn an_unmigrated_legacy_record_declines_with_exit_2_and_is_named() {
    let r = shapes_on("parity-unmigrated");
    assert_eq!(r.code, 2, "a refusal is a DECLINE, not a Fail\n{}", r.all());
    assert!(r.stderr.contains("UNMIGRATED"), "{}", r.all());
    assert!(
        r.stderr.contains("legacy.json"),
        "named by file\n{}",
        r.all()
    );
}

#[test]
fn a_receipt_added_without_bumping_the_denominator_declines_naming_both_numbers() {
    // The falsifier the row names: the extractor found 2, the committed denominator says 1.
    let r = shapes_on("parity-denominator-drift");
    assert_eq!(r.code, 2, "{}", r.all());
    assert!(r.stderr.contains("matched 2"), "{}", r.all());
    assert!(r.stderr.contains("says 1"), "{}", r.all());
}

#[test]
fn the_three_answers_are_distinct() {
    // A build that collapses decline into fail, or fail into pass, fails here rather than in a release.
    let codes: Vec<i32> = ["parity-green", "parity-nocomparator", "parity-unmigrated"]
        .iter()
        .map(|f| shapes_on(f).code)
        .collect();
    assert_eq!(codes, vec![0, 1, 2], "pass / fail / decline must differ");
}

#[test]
fn the_entity_type_is_counted_in_by_entity_type_not_merely_registered() {
    // ONT-001 v4.10's probe asks `by_entity_type["parity-receipt"] == 7`. Registering the entity
    // type in Σ is not enough for that: without a key here the probe reads ABSENT, and an absent
    // key is not zero — a consumer treating it as one measures nothing and calls it a pass. Same
    // shape as #3610, one map over.
    let out = Command::new(pv_bin())
        .args(["lint", "contracts", "--gate", "shapes", "--format", "json"])
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn pv");
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).expect("json report");
    let counted = &v["extra"]["by_entity_type"]["parity-receipt"];
    assert!(
        !counted.is_null(),
        "by_entity_type carries no `parity-receipt` key — the probe would read ABSENT, not 7"
    );
    assert_eq!(
        counted.as_u64(),
        v["extra"]["by_shape"]
            .as_array()
            .expect("by_shape")
            .iter()
            .find_map(|s| s
                .as_str()?
                .strip_prefix("parity-receipt-complete=")?
                .parse()
                .ok()),
        "the entity count and the shape's focus-node count must be the same number"
    );
}

#[test]
fn the_parity_extractor_control_is_drawn_by_the_gate_every_run() {
    // PMAT-3704. ONT-001 v4.10's probe also asks `.pc_extract["parity-receipt"] == "fired"`, at the TOP level
    // of the report, exactly as read here. The control existed from #3600 on — `parity_receipt::positive_control`
    // — but only a unit test called it, so the gate never reported it and an extractor that stopped reading the
    // v2 layout would have been counted, never refused. R-3: one planted defect per registered extractor, every
    // run.
    let out = Command::new(pv_bin())
        .args(["lint", "contracts", "--gate", "shapes", "--format", "json"])
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn pv");
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).expect("json report");
    assert_eq!(
        v["pc_extract"]["parity-receipt"], "fired",
        "pc_extract carries no fired `parity-receipt` control: {}",
        v["pc_extract"]
    );
}

#[test]
fn every_fixture_carries_the_real_contract_byte_for_byte() {
    // A fixture copy that drifts from `contracts/parity-receipt-v2.yaml` would let the real shape be mutated
    // while the case table stayed green — the mutation control's blind spot, closed here.
    let real = std::fs::read(repo_root().join("contracts/parity-receipt-v2.yaml"))
        .expect("the real contract is in the tree");
    for name in [
        "parity-green",
        "parity-nocomparator",
        "parity-unknownkind",
        "parity-unmigrated",
        "parity-denominator-drift",
    ] {
        let copy = std::fs::read(fixture(name).join("contracts/parity-receipt-v2.yaml"))
            .unwrap_or_else(|e| panic!("{name} carries the contract: {e}"));
        assert_eq!(
            copy, real,
            "{name}'s copy of parity-receipt-v2.yaml has drifted from contracts/"
        );
    }
}

#[test]
fn the_committed_tree_agrees_with_its_own_denominator() {
    // The real corpus, not a fixture: the independent predicate and the extractor must reach the same count.
    let out = Command::new("bash")
        .arg("scripts/parity_receipt_denominator.sh")
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn the denominator predicate");
    // #3694: the classifier is awk, so a python-free runner (infra#708) measures the count like any
    // other. The UNMEASURED exit 3 that #3695 accepted here as a stop-gap is gone: anything but
    // success is RED.
    assert!(
        out.status.success(),
        "exit {:?}\n{}{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
