//! ONT-8 (PMAT-4077) — `pv lint --gate evidence`: one evidence block, PROV-O names, one L-enum, every entity type.
//!
//! The row's probe, verbatim (paiml/infra `docs/specifications/paiml-ontology.md` v4.14 :667):
//! `pv lint contracts/ --gate evidence --format json … jq -e '.verdict=="Pass" and .levels_source=="enum" and
//! .entity_types_checked>=3'`. The first test is that probe on the real corpus.
//!
//! The rest build a corpus in a tempdir around the repo's own Σ, so every rule is judged against the agents and
//! levels the corpus really declares:
//!
//! | corpus | expected |
//! |---|---|
//! | readme + apr-model + code, one identical block | exit 0, Pass, `entity_types_checked` 3 — R-17 |
//! | `author:` under `provenance` | exit 1, PV-ONT-017 — F-16 |
//! | `level: L0` | exit 1, PV-ONT-018 — EV-3's one enum has no L0 |
//! | `wasAttributedTo: orchestrator` | exit 1, PV-ONT-021 — not a Σ agent |
//! | no evidence block anywhere | exit 2, decline — R-2 |
//! | no Σ | exit 2, decline |
//!
//! DISCRIMINATION: the R-17 corpus passes at exit 0 and every reject names its own rule id, so a build that
//! refuses everything, or refuses with the wrong rule, fails this file.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
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

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn gate(dir: &Path) -> Run {
    pv(&[
        "lint",
        dir.to_str().expect("utf-8 path"),
        "--gate",
        "evidence",
        "--format",
        "json",
    ])
}

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

const FULL: &str = "level: L2\nmark: C\nprovenance:\n  wasGeneratedBy: {command: \"pv extract README.md\"}\n  wasAttributedTo: pv\n  generatedAtTime: \"2026-09-24T00:00:00Z\"\n";

fn contract(entity_type: &str, evidence: &str) -> String {
    let indented: String = evidence.lines().map(|l| format!("  {l}\n")).collect();
    format!(
        "metadata:\n  version: \"1.0.0\"\nentity:\n  type: {entity_type}\nevidence:\n{indented}"
    )
}

/// A corpus around the repo's Σ. `with_sigma: false` leaves Σ out.
fn corpus(with_sigma: bool, files: &[(&str, String)]) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("corpus dir is creatable");
    if with_sigma {
        std::fs::copy(
            repo_contracts().join("ontology.yaml"),
            tmp.path().join("ontology.yaml"),
        )
        .expect("Σ copies");
    }
    for (name, body) in files {
        std::fs::write(tmp.path().join(name), body).expect("contract writes");
    }
    tmp
}

/// The rejects: exit 1, verdict Fail, and the report names exactly this rule.
fn assert_rejects(evidence: &str, rule: &str) {
    let tmp = corpus(true, &[("c-v1.yaml", contract("code", evidence))]);
    let r = gate(tmp.path());
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Fail", "{}", show(&r));
    let rules: Vec<_> = v["findings"]
        .as_array()
        .expect("findings is an array")
        .iter()
        .map(|f| f["rule_id"].clone())
        .collect();
    assert_eq!(
        rules,
        [serde_json::Value::String(rule.into())],
        "{}",
        show(&r)
    );
}

#[test]
fn the_row_probe_passes_on_the_real_corpus() {
    let r = gate(&repo_contracts());
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["gate"], "evidence", "{}", show(&r));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["levels_source"], "enum", "{}", show(&r));
    assert!(
        v["entity_types_checked"]
            .as_u64()
            .expect("entity_types_checked is a number")
            >= 3,
        "{}",
        show(&r)
    );
}

#[test]
fn r17_readme_model_and_code_contracts_pass_the_same_gate() {
    let tmp = corpus(
        true,
        &[
            ("readme-v1.yaml", contract("readme", FULL)),
            ("model-v1.yaml", contract("apr-model", FULL)),
            ("code-v1.yaml", contract("code", FULL)),
        ],
    );
    let r = gate(tmp.path());
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["entity_types_checked"], 3, "{}", show(&r));
    assert_eq!(v["contracts_with_evidence"], 3, "{}", show(&r));
}

#[test]
fn f16_author_is_rejected() {
    assert_rejects(
        &FULL.replace(
            "  wasAttributedTo: pv\n",
            "  wasAttributedTo: pv\n  author: noah\n",
        ),
        "PV-ONT-017",
    );
}

#[test]
fn level_l0_is_rejected() {
    assert_rejects(&FULL.replace("L2", "L0"), "PV-ONT-018");
}

#[test]
fn an_undeclared_agent_is_rejected() {
    assert_rejects(
        &FULL.replace("wasAttributedTo: pv", "wasAttributedTo: orchestrator"),
        "PV-ONT-021",
    );
}

#[test]
fn no_evidence_anywhere_is_a_decline() {
    let tmp = corpus(
        true,
        &[("c-v1.yaml", "metadata:\n  version: \"1.0.0\"\n".to_string())],
    );
    let r = gate(tmp.path());
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("no evidence block"), "{}", show(&r));
}

#[test]
fn no_sigma_is_a_decline() {
    let tmp = corpus(false, &[("c-v1.yaml", contract("code", FULL))]);
    let r = gate(tmp.path());
    assert_eq!(r.code, 2, "{}", show(&r));
}

/// R-8: computed in every `pv lint` run, and reported as not armed until the baseline names it.
#[test]
fn a_full_run_computes_the_gate() {
    let r = pv(&[
        "lint",
        repo_contracts().to_str().expect("utf-8 path"),
        "--format",
        "json",
    ]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    let g = v["gates"]
        .as_array()
        .expect("gates is an array")
        .iter()
        .find(|g| g["name"] == "evidence")
        .unwrap_or_else(|| panic!("no evidence gate in a full run\n{}", show(&r)));
    assert_eq!(g["passed"], true, "{}", show(&r));
}
