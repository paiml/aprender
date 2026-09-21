//! ONT-4b (PMAT-3500) — `pv lint --gate shapes` and `pv extract`: the first usable shapes gate.
//!
//! ONT-001 v4.5 §5 ONT-4b RED, verbatim: "zero blank nodes, two extractions byte-identical; a shape using an
//! unsupported component → exit 3 `error: shape uses unsupported <x>`; `shapes_n==0 → Unknown{NoShapes}`; empty
//! data → `Unknown{NoFocus}`; `pc_shape` yields exactly one violation; a corpus violation → reject naming focus node
//! and shape; warnings only → `Unknown{Warn}`; `contracts/shapes.ttl` exported from the `shape:` blocks."
//!
//! DISCRIMINATION: `shapes-ok/` must PASS at exit 0 with the plant fired, so a build that declines every corpus
//! fails this file; and the real corpus must pass with more than a thousand focus nodes, so a build whose extractor
//! emits nothing fails it too.

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

fn pv_in(cwd: &Path, args: &[&str]) -> Run {
    let out = Command::new(pv_bin())
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    pv_in(scratch.path(), args)
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
    pv(&["lint", &s(dir), "--gate", "shapes", "--format", "json"])
}

#[test]
fn the_repo_corpus_passes_with_the_plant_fired_and_the_whole_corpus_as_focus_nodes() {
    let r = gate(&repo_contracts());
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["gate"], "shapes");
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert!(
        v["extra"]["shapes_n"].as_u64().unwrap_or(0) >= 1,
        "{}",
        show(&r)
    );
    assert!(
        v["extra"]["focus_nodes_n"].as_u64().unwrap_or(0) > 1000,
        "every contract's projection is a focus node\n{}",
        show(&r)
    );
    assert_eq!(v["extra"]["pc_shape"], "fired", "{}", show(&r));
    // ONT-4b: exactly one, from ont:id minCount. ONT-4c1 plants a bare model:Model too, which draws
    // ladder-measured's two minCounts: three on this corpus, and never zero.
    assert_eq!(
        v["extra"]["plant_violations"],
        3,
        "ont:id minCount + ladder-measured's two (ONT-4c1)\n{}",
        show(&r)
    );
}

#[test]
fn a_conforming_fixture_passes_at_exit_0() {
    let r = gate(&fixture("shapes-ok"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass");
    assert_eq!(
        v["extra"]["focus_nodes_n"],
        3,
        "the plant is not counted\n{}",
        show(&r)
    );
}

#[test]
fn a_corpus_violation_rejects_at_exit_1_naming_focus_node_and_shape() {
    let r = gate(&fixture("shapes-violation"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert_eq!(json_of(&r)["verdict"], "Fail");
    assert!(
        r.stdout.contains("ont:contract/bad-KIND"),
        "the focus node\n{}",
        show(&r)
    );
    assert!(
        r.stdout.contains("shape `shape-holder`"),
        "the shape\n{}",
        show(&r)
    );
    assert!(r.stdout.contains("PV-ONT-011"), "{}", show(&r));
}

#[test]
fn warnings_only_declines_at_exit_2_with_warn() {
    let r = gate(&fixture("shapes-warn"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: Warn"), "{}", show(&r));
}

#[test]
fn no_shapes_declines_at_exit_2() {
    let r = gate(&fixture("sigma-ok"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: NoShapes"), "{}", show(&r));
}

#[test]
fn no_focus_node_declines_at_exit_2() {
    let r = gate(&fixture("shapes-nofocus"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: NoFocus"), "{}", show(&r));
}

#[test]
fn a_dead_plant_declines_at_exit_2_never_passes() {
    let r = gate(&fixture("shapes-noplant"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(
        r.stderr.contains("decline: PositiveControlFailed"),
        "{}",
        show(&r)
    );
}

#[test]
fn an_unsupported_component_is_an_error_at_exit_3_naming_it() {
    let r = gate(&fixture("shapes-unsupported"));
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.starts_with("error: "), "{}", show(&r));
    assert!(r.stderr.contains("unsupported targetNode"), "{}", show(&r));
}

/// R-15: two extractions are byte-identical and carry no blank node; R-18: `--check` sees drift.
#[test]
fn extract_is_deterministic_and_extract_check_sees_drift() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("contracts");
    // copy a fixture corpus into a scratch dir so the test may write next to it
    std::fs::create_dir_all(&dir).expect("scratch contracts dir");
    for e in std::fs::read_dir(fixture("shapes-ok")).expect("fixture dir") {
        let p = e.expect("fixture entry").path();
        std::fs::copy(&p, dir.join(p.file_name().expect("file name"))).expect("copy fixture");
    }
    let d = s(&dir);
    let a = pv(&["extract", &d]);
    assert_eq!(a.code, 0, "{}", show(&a));
    let nt1 = std::fs::read_to_string(dir.join("contracts.nt")).expect("contracts.nt written");
    assert!(!nt1.contains("_:"), "no blank nodes");
    assert!(nt1.contains("/contract/shape-holder> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://ont.paiml.dev/v1alpha1/Contract> ."));
    let ttl = std::fs::read_to_string(dir.join("shapes.ttl")).expect("shapes.ttl written");
    assert!(
        ttl.contains("sh:NodeShape") && ttl.contains("sh:minCount 1"),
        "{ttl}"
    );
    let b = pv(&["extract", &d]);
    assert_eq!(b.code, 0);
    let nt2 = std::fs::read_to_string(dir.join("contracts.nt")).expect("contracts.nt rewritten");
    assert_eq!(nt1, nt2, "byte-identical");
    assert_eq!(json_of(&a)["sha256"], json_of(&b)["sha256"]);
    let ok = pv(&["extract", &d, "--check"]);
    assert_eq!(ok.code, 0, "{}", show(&ok));
    // drift: a contract changes, the tracked file does not
    std::fs::write(dir.join("other-c.yaml"), "metadata:\n  version: '1.0.0'\n  kind: kernel\n  created: '2026-09-18'\n  last_modified: '2026-09-18'\n  author: fixture\n  description: fixture\nname: other-c\nversion: '1.0.0'\nscope: fixture\nstatus: active\n").expect("write other-c");
    let drift = pv(&["extract", &d, "--check"]);
    assert_eq!(drift.code, 1, "{}", show(&drift));
    assert!(
        drift.stderr.contains("differs from a fresh extraction"),
        "{}",
        show(&drift)
    );
}

/// The tracked graph in this repository IS a fresh extraction (R-18), and `shapes.ttl` is the export.
#[test]
fn the_tracked_repo_graph_is_fresh() {
    let r = pv(&["extract", &s(&repo_contracts()), "--check"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert!(v["triples"].as_u64().unwrap_or(0) > 5000, "{}", show(&r));
    // MEASURED on this branch: `pv extract contracts --check` reports
    // shapes_n=9, triples=15851 (the 0.69 batch, #3669, pinned pv 0.68.2 built from
    // the tree; was 6 / 15620 before #3600 folded in). The count is deliberately hardcoded rather
    // than derived — a shape added without anyone noticing is the thing this
    // assertion exists to prevent, so adding one is SUPPOSED to turn it red and
    // make you name the new shape here.
    //
    // It did exactly that for #3605's `refusal-receipt-v1`, and a quorum lane
    // caught it rather than I did. Note for whoever merges second: this counter
    // is shared across branches, so a sibling PR that also adds a shape
    // (#3600's parity-receipt-v2) will need the number raised again at merge —
    // that is the ratchet working, not a conflict to route around. It did: the
    // 0.69 batch folded #3600 in and the count went 6 -> 9 with its three shapes.
    assert_eq!(
        v["shapes_n"],
        9,
        "ont-shapes-v1 + ladder-measured + ladder-green (ONT-4c1) + bound-symbols-resolve + lean-statements-grounded (ONT-4b2) + refusal-receipt-v1 (#3605) + parity-receipt-complete + parity-comparator-self + parity-comparator-oracle (parity-receipt-v2, #3600)\n{}",
        show(&r)
    );
}

/// R-8: the gate is computed in every `pv lint` run.
#[test]
fn the_full_run_computes_the_shapes_gate() {
    let r = pv(&["lint", &s(&repo_contracts()), "--format", "json"]);
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
        names.iter().any(|n| n == "shapes"),
        "{names:?}\n{}",
        show(&r)
    );
}
