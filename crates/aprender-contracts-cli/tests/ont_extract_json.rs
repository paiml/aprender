//! `extract:json` (aprender#3515, PMAT-3515; ONT-001 §3.7) — a tool's own `--json` output as a shaped entity under
//! `pv lint --gate shapes`, for paiml/infra's ARBITER-001 §14 (every `arbiter … --json` is a contract whose CLOSED
//! shape is the interface definition).
//!
//! The falsifiers, verbatim from the issue: conforming → Pass with the plant fired; an undeclared key against a
//! closed shape → Fail naming focus and shape; a value outside an `in:` list → Fail; an unmapped nested key →
//! exit 3 naming it; a missing ref → exit 3 naming the path; a torn JSONL line → `Unknown{Warn}` naming the line
//! with the other nodes present; two extractions byte-identical.
//!
//! DISCRIMINATION: `json-ok/` must PASS at exit 0 with `focus_nodes_n == 1`, so a build whose gate never sees the
//! JSON entity (the state before this row: `Unknown{NoFocus}`) fails this file.

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
        .join("contracts")
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
fn a_conforming_json_entity_passes_with_the_plant_fired_and_one_focus_node() {
    let r = gate(&fixture("json-ok"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let j = json_of(&r);
    assert_eq!(j["verdict"], "Pass", "{}", show(&r));
    assert_eq!(j["extra"]["focus_nodes_n"], 1, "{}", show(&r));
    assert_eq!(j["extra"]["pc_shape"], "fired", "{}", show(&r));
    assert_eq!(j["extra"]["violations"], 0, "{}", show(&r));
    assert!(
        j["extra"]["triples"].as_u64().unwrap_or(0) > 20,
        "the entity's nodes are in the graph\n{}",
        show(&r)
    );
}

#[test]
fn an_undeclared_key_and_a_value_outside_in_are_each_a_violation_naming_focus_and_shape() {
    let r = gate(&fixture("json-violation"));
    assert_eq!(r.code, 1, "{}", show(&r));
    let j = json_of(&r);
    assert_eq!(j["verdict"], "Fail", "{}", show(&r));
    assert_eq!(j["extra"]["violations"], 2, "{}", show(&r));
    assert!(
        r.stdout.contains("ont:tool/tool-status"),
        "the focus node\n{}",
        show(&r)
    );
    assert!(
        r.stdout.contains("shape `tool-status`"),
        "the shape\n{}",
        show(&r)
    );
    assert!(
        r.stdout.contains("(closed)") && r.stdout.contains("ont:tool/surprise"),
        "the closed violation names the undeclared key\n{}",
        show(&r)
    );
    assert!(
        r.stdout.contains("broken is not one of"),
        "the in: violation names the value\n{}",
        show(&r)
    );
}

#[test]
fn an_unmapped_nested_key_is_exit_3_naming_the_key() {
    let r = gate(&fixture("json-unmapped"));
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(
        r.stderr
            .contains("nested key `findings` is not in vocabulary.nested"),
        "{}",
        show(&r)
    );
}

#[test]
fn a_missing_ref_is_exit_3_naming_the_path() {
    let r = gate(&fixture("json-missing-ref"));
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(
        r.stderr
            .contains("entity.ref `data/does-not-exist.json` cannot be read"),
        "{}",
        show(&r)
    );
}

#[test]
fn a_torn_jsonl_line_declines_with_warn_naming_the_line_and_the_other_rows_are_focus_nodes() {
    let r = gate(&fixture("json-torn-jsonl"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: Warn"), "{}", show(&r));
    let j = json_of(&r);
    assert_eq!(j["verdict"], "Unknown(Warn)", "{}", show(&r));
    assert_eq!(
        j["extra"]["focus_nodes_n"],
        3,
        "line 2 (NUL-prefixed) salvaged, line 3 not\n{}",
        show(&r)
    );
    assert_eq!(j["extra"]["warnings"], 1, "{}", show(&r));
    assert!(
        r.stdout.contains("line 3 unparsable"),
        "the warning names the line\n{}",
        show(&r)
    );
}

#[test]
fn pv_extract_carries_the_json_entity_and_is_deterministic() {
    let dir = tempfile::tempdir().expect("scratch");
    let src = fixture("json-ok");
    let root = src.parent().expect("fixture root");
    // Copy the fixture (contracts/ + data/) so `pv extract` may write contracts.nt beside it.
    for sub in ["contracts", "data"] {
        let to = dir.path().join(sub);
        std::fs::create_dir_all(&to).expect("scratch subdir");
        for e in std::fs::read_dir(root.join(sub)).expect("fixture subdir") {
            let e = e.expect("dir entry");
            std::fs::copy(e.path(), to.join(e.file_name())).expect("copy fixture file");
        }
    }
    let d = s(&dir.path().join("contracts"));
    let a = pv(&["extract", &d]);
    assert_eq!(a.code, 0, "{}", show(&a));
    let nt1 = std::fs::read_to_string(dir.path().join("contracts/contracts.nt"))
        .expect("contracts.nt written");
    assert!(
        nt1.contains("ont.paiml.dev/v1alpha1/tool/Status"),
        "the JSON root's class\n{nt1}"
    );
    assert!(
        nt1.contains("tool/tool-status.push>"),
        "the nested node\n{nt1}"
    );
    assert!(!nt1.contains("_:"), "no blank nodes\n{nt1}");
    let b = pv(&["extract", &d]);
    assert_eq!(b.code, 0, "{}", show(&b));
    let nt2 = std::fs::read_to_string(dir.path().join("contracts/contracts.nt"))
        .expect("contracts.nt rewritten");
    assert_eq!(nt1, nt2, "two extractions are byte-identical");
    let ok = pv(&["extract", &d, "--check"]);
    assert_eq!(ok.code, 0, "{}", show(&ok));
}

#[test]
fn pv_extract_on_an_unmapped_key_is_exit_3_not_a_partial_graph() {
    let r = pv(&["extract", &s(&fixture("json-unmapped")), "--check"]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.contains("nested key `findings`"), "{}", show(&r));
}
