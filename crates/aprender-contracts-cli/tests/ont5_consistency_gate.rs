//! ONT-5 (PMAT-4074) — `pv-sat` writes the witness, `pv lint --gate ont-consistency` checks it, on the CLI.
//!
//! ONT-001 §5 ONT-5 probe, verbatim in `the_spec_probe_holds_on_the_repo_corpus`. Around it, every exit the gate
//! can give: 0 on the repo's consistent relations, 1 (PV-ONT-022) on `relations-ok` — where `a contradicts d` and
//! both are live, so a gate that never reasons would pass it — 2 for no witness / no Σ / no relation, and 3 for a
//! Σ that does not parse. The witness in every non-repo case is written by the real `pv-sat` binary, so the two
//! halves are tested against each other, not against a hand-built file.

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

fn pv(args: &[&str]) -> Run {
    run(env!("CARGO_BIN_EXE_pv"), args)
}

fn pv_sat(dir: &Path) -> Run {
    run(env!("CARGO_BIN_EXE_pv-sat"), &[s(dir)])
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

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

fn gate(dir: &Path) -> Run {
    pv(&[
        "lint",
        s(dir),
        "--gate",
        "ont-consistency",
        "--format",
        "json",
    ])
}

/// A copy of `tests/fixtures/ont/relations-ok`.
fn corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for entry in std::fs::read_dir(repo().join("tests/fixtures/ont/relations-ok")).expect("fixture")
    {
        let entry = entry.expect("entry");
        std::fs::copy(entry.path(), dir.path().join(entry.file_name())).expect("copy");
    }
    dir
}

#[test]
fn the_spec_probe_holds_on_the_repo_corpus() {
    // F-7: the reasoner is not reachable from the library.
    let lib = std::fs::read_to_string(repo().join("crates/aprender-contracts/src/lib.rs"))
        .expect("lib.rs");
    assert!(!lib.contains("mod sat"), "the lib reaches the reasoner");
    let r = gate(&repo().join("contracts"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert!(
        v["checkable_n"].as_u64().is_some_and(|n| n > 0),
        "{}",
        show(&r)
    );
    assert_eq!(v["pc_checker"], "fired", "{}", show(&r));
    assert_eq!(v["witness"]["pc_reasoner"], "fired", "{}", show(&r));
    assert_eq!(v["witness"]["stale"], false, "{}", show(&r));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
}

#[test]
fn an_inconsistent_corpus_fails_on_the_witness_pv_sat_wrote() {
    let d = corpus();
    let w = pv_sat(d.path());
    assert_eq!(w.code, 0, "{}", show(&w));
    assert!(w.stdout.contains("unsat_core"), "{}", show(&w));
    let r = gate(d.path());
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-ONT-022"), "{}", show(&r));
    assert_eq!(
        json_of(&r)["core"],
        serde_json::Value::from(vec!["a", "d"]),
        "{}",
        show(&r)
    );
}

#[test]
fn an_edit_after_the_witness_is_stale_and_says_make_contracts() {
    let d = corpus();
    assert_eq!(pv_sat(d.path()).code, 0);
    let a = d.path().join("a.yaml");
    let text = std::fs::read_to_string(&a).expect("a.yaml");
    std::fs::write(&a, text.replace("  contradicts: [d]\n", "")).expect("edit");
    let r = gate(d.path());
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("make contracts"), "{}", show(&r));
    assert!(r.stderr.contains("WitnessStale"), "{}", show(&r));

    // Regenerating is the whole remedy: the edited corpus is consistent, and says so.
    assert_eq!(pv_sat(d.path()).code, 0);
    let r = gate(d.path());
    assert_eq!(r.code, 0, "{}", show(&r));
    let files = std::fs::read_dir(d.path().join("witness"))
        .expect("witness dir")
        .count();
    assert_eq!(files, 1, "the old graph's witness was not pruned");
}

#[test]
fn nothing_to_check_declines_and_a_broken_sigma_is_exit_3() {
    let d = corpus();
    for name in ["a.yaml", "b.yaml"] {
        let p = d.path().join(name);
        let text = std::fs::read_to_string(&p).expect("read");
        let cut = text.find("relations:").expect("relations block");
        std::fs::write(&p, &text[..cut]).expect("write");
    }
    let r = gate(d.path());
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("NoCheckable"), "{}", show(&r));
    assert_eq!(
        pv_sat(d.path()).code,
        2,
        "pv-sat must not write a witness for zero clauses"
    );

    let d = corpus();
    std::fs::remove_file(d.path().join("ontology.yaml")).expect("rm");
    assert_eq!(gate(d.path()).code, 2);

    let d = corpus();
    std::fs::write(d.path().join("ontology.yaml"), "roles: [unclosed\n").expect("write");
    let r = gate(d.path());
    assert_eq!(r.code, 3, "{}", show(&r));
}

/// R-8: computed in every `pv lint` run, and not armed until the cop arms it.
#[test]
fn the_full_run_computes_the_gate_unarmed() {
    let r = pv(&["lint", s(&repo().join("contracts")), "--format", "json"]);
    let v = json_of(&r);
    let gate = v["gates"]
        .as_array()
        .and_then(|g| g.iter().find(|x| x["name"] == "ont-consistency"))
        .unwrap_or_else(|| panic!("ont-consistency not computed\n{}", show(&r)));
    assert_eq!(gate["verdict"], "Pass", "{gate}");
    let not_armed: Vec<&str> = v["not_armed"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    assert!(not_armed.contains(&"ont-consistency"), "{not_armed:?}");
}
