//! ONT-4e (PMAT-4075) — `refines` is Liskov or it is rejected (R-20), on the CLI.
//!
//! ONT-001 §5 ONT-4e probe, verbatim in `the_spec_probe_holds_on_the_repo_corpus`. Around it, each RED line of the
//! row: a strengthened precondition and a weakened postcondition are `reject:` lines naming the clause, a dropped
//! invariant is exit 1, a `prose` clause is `Unknown{Prose}` naming the clause, a legacy `invariants[]`-only pair is
//! `Pass` with `liskov_pairs_checked: 0` on the verdict line. Every witness is written by the real `pv-sat` binary, so
//! the reasoner and the checker are tested against each other; a tampered witness is PV-ONT-025, never a verdict.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(bin: &str, args: &[&str]) -> Run {
    let out = Command::new(bin).args(args).output().expect("spawn");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

fn show(r: &Run) -> String {
    format!(
        "exit {}\n--- stdout\n{}\n--- stderr\n{}",
        r.code, r.stdout, r.stderr
    )
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn pv_sat(dir: &Path) -> Run {
    run(env!("CARGO_BIN_EXE_pv-sat"), &[s(dir)])
}

fn gate(dir: &Path) -> Run {
    run(
        env!("CARGO_BIN_EXE_pv"),
        &["lint", s(dir), "--gate", "refines", "--format", "json"],
    )
}

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

/// A copy of `tests/fixtures/ont/<name>`.
fn fixture(name: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for entry in std::fs::read_dir(repo().join("tests/fixtures/ont").join(name)).expect("fixture") {
        let entry = entry.expect("entry");
        std::fs::copy(entry.path(), dir.path().join(entry.file_name())).expect("copy");
    }
    dir
}

/// The fixture, with the witness the real pv-sat wrote for it, gated.
fn reasoned(name: &str) -> (tempfile::TempDir, Run) {
    let dir = fixture(name);
    let sat = pv_sat(dir.path());
    assert_eq!(sat.code, 0, "pv-sat on {name}: {}", show(&sat));
    let r = gate(dir.path());
    (dir, r)
}

fn liskov_witnesses(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir.join("witness/liskov"))
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default()
}

#[test]
fn the_spec_probe_holds_on_the_repo_corpus() {
    // ONT-001 §5 ONT-4e probe: `"$PV" lint contracts/ --gate refines --format json >"$TMP/lk.json"
    // && json_object "$TMP/lk.json" && jq -e '.liskov_pairs_checked>0 and .pc_checker=="fired" and .verdict=="Pass"'`
    let r = gate(&repo().join("contracts"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert!(v.is_object(), "json_object: {v}");
    assert!(
        v["liskov_pairs_checked"].as_u64().unwrap_or(0) > 0,
        "liskov_pairs_checked>0: {v}"
    );
    assert_eq!(v["pc_checker"], "fired", "{v}");
    assert_eq!(v["verdict"], "Pass", "{v}");
    assert_eq!(v["witness"]["pc_reasoner"], "fired", "{v}");
}

#[test]
fn a_liskov_pair_passes_and_the_witness_is_a_chain() {
    let (dir, r) = reasoned("refines-ok");
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{v}");
    assert_eq!(v["liskov_pairs_checked"], 1, "{v}");
    assert_eq!(v["violations"], 0, "{v}");
    let w = liskov_witnesses(dir.path());
    assert_eq!(w.len(), 1, "one Liskov witness: {w:?}");
    let text = std::fs::read_to_string(&w[0]).expect("witness");
    assert!(text.contains("\"chain\""), "{text}");
    assert!(!text.contains("counter_model"), "{text}");

    // Byte-stable: a second pv-sat run confirms, it does not rewrite.
    let again = pv_sat(dir.path());
    assert_eq!(again.code, 0, "{}", show(&again));
    assert_eq!(
        std::fs::read_to_string(&w[0]).expect("witness"),
        text,
        "rerun rewrote the witness"
    );
}

#[test]
fn a_strengthened_precondition_is_rejected_naming_the_clause() {
    let (_dir, r) = reasoned("refines-pre-strengthened");
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stderr
            .contains("reject: a refines b: precondition strengthened (PRE-1)"),
        "{}",
        show(&r)
    );
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Fail", "{v}");
    assert_eq!(v["findings"][0]["rule_id"], "PV-ONT-024", "{v}");
}

#[test]
fn a_weakened_postcondition_is_rejected_naming_the_clause() {
    let (_dir, r) = reasoned("refines-post-weakened");
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stderr
            .contains("reject: a refines b: postcondition weakened (POST-1)"),
        "{}",
        show(&r)
    );
}

#[test]
fn a_dropped_invariant_is_rejected() {
    let (_dir, r) = reasoned("refines-inv-dropped");
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stderr
            .contains("reject: a refines b: invariant dropped (INV-1)"),
        "{}",
        show(&r)
    );
}

#[test]
fn a_prose_clause_is_unknown_prose_naming_the_clause() {
    let (_dir, r) = reasoned("refines-prose");
    assert_eq!(r.code, 2, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Unknown(Prose)", "{v}");
    assert_eq!(v["liskov_pairs_checked"], 0, "{v}");
    assert_eq!(v["liskov_prose"], 1, "{v}");
    assert_eq!(v["prose_clauses"][0], "a.requires PRE-1", "{v}");
    assert!(r.stderr.contains("a.requires PRE-1"), "{}", show(&r));
    assert!(r.stderr.contains("decline: Prose"), "{}", show(&r));
}

#[test]
fn a_legacy_pair_passes_and_says_nothing_was_checked() {
    // No witness needed: nothing to check, and the verdict line says so.
    let dir = fixture("refines-legacy");
    let r = gate(dir.path());
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{v}");
    assert_eq!(v["liskov_pairs_checked"], 0, "{v}");
    assert_eq!(v["liskov_pairs_legacy"], 1, "{v}");
    assert_eq!(v["pc_checker"], "fired", "{v}");
}

#[test]
fn no_witness_is_a_stale_decline_naming_make_contracts() {
    let dir = fixture("refines-ok");
    let r = gate(dir.path());
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("make contracts"), "{}", show(&r));
    assert!(r.stderr.contains("decline: WitnessStale"), "{}", show(&r));
}

#[test]
fn a_tampered_witness_is_refused_not_believed() {
    // pv-sat proves the pre-strengthened pair violates PRE-1; a hand edit turns the counter-model into a chain
    // that derives A's PRE-1 from nothing B requires. The checker must refuse it (PV-ONT-025), not pass it.
    let dir = fixture("refines-pre-strengthened");
    assert_eq!(pv_sat(dir.path()).code, 0);
    let w = liskov_witnesses(dir.path());
    assert_eq!(w.len(), 1, "{w:?}");
    let mut doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&w[0]).expect("witness")).expect("json");
    let obligations = doc["pairs"][0]["obligations"]
        .as_array_mut()
        .expect("obligations");
    for ob in obligations.iter_mut() {
        if ob["kind"] == "pre" {
            *ob = serde_json::from_str(
                r#"{"kind": "pre", "chain": [{"clause": "PRE-1", "from": "PRE-1"}]}"#,
            )
            .expect("obligation");
        }
    }
    std::fs::write(&w[0], serde_json::to_string_pretty(&doc).expect("ser")).expect("write");
    let r = gate(dir.path());
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["findings"][0]["rule_id"], "PV-ONT-025", "{v}");
    assert!(
        !r.stderr.contains("precondition strengthened"),
        "a refused witness names no verdict: {}",
        show(&r)
    );
}

#[test]
fn pv_sat_self_test_fires_the_liskov_controls() {
    let r = run(env!("CARGO_BIN_EXE_pv-sat"), &["--self-test"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    assert!(
        r.stdout.contains("pc_liskov_reasoner fired"),
        "{}",
        show(&r)
    );
    assert!(r.stdout.contains("pc_liskov_checker fired"), "{}", show(&r));
}
