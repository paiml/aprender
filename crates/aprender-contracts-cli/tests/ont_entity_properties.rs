//! PMAT-3529 — the two defects apex measured on `pv 0.68.1`, on the CLI.
//!
//! 1. Σ carries the CONTRACT schema's `metadata:` block. `pv validate` requires `metadata.{version,
//!    description}` on a contract and `contracts/ontology.yaml` IS a contract, so before this one file could
//!    not satisfy both readers: `pv validate` rc 0 and `pv lint --gate sigma` exit 3 over the same bytes.
//! 2. `entity.properties.<k>` becomes `<entityType>:<k>` on the contract node, so a `shape:` can constrain the
//!    entity's own properties — apex's nine pre-registered degrees of freedom, which live in the contract
//!    because the contract IS the pre-registration (APEX-001 EV-6 hashes it).
//!
//! The three fixtures are apex's EV-21 falsifiers: conforming, a value outside `in:`, and a tenth key under
//! `closed: true`. DISCRIMINATION: `entity-props-ok/` must PASS (a build that emits no property triples fails
//! it on `minCount`), and the two others must FAIL naming the PROPERTY (a build that emits them under `ont:`
//! or drops the path from the message fails those).

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

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

fn gate(name: &str, which: &str) -> Run {
    pv_in(
        &fixture(name),
        &["lint", "contracts", "--gate", which, "--format", "json"],
    )
}

fn messages(v: &serde_json::Value) -> Vec<String> {
    v["findings"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|f| f["message"].as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn the_sigma_gate_accepts_the_contract_schemas_metadata_block_that_validate_requires() {
    // The same file, both readers: this is the contradiction the ticket exists to end.
    let dir = fixture("entity-props-ok");
    let sigma = gate("entity-props-ok", "sigma");
    assert_eq!(
        sigma.code,
        0,
        "the sigma gate reads Σ with metadata\n{}",
        show(&sigma)
    );
    let validate = pv_in(&dir, &["validate", "contracts/ontology.yaml"]);
    assert_eq!(
        validate.code,
        0,
        "pv validate accepts the same file\n{}",
        show(&validate)
    );
}

#[test]
fn an_unclaimed_metadata_block_is_still_refused_by_name() {
    // The anti-decoration rule is answered, not weakened: drop the `readers` entry and Σ is malformed again.
    let scratch = tempfile::tempdir().expect("scratch");
    let src = fixture("entity-props-ok").join("contracts");
    let dst = scratch.path().join("contracts");
    std::fs::create_dir_all(&dst).expect("mkdir");
    for entry in std::fs::read_dir(&src).expect("read fixture").flatten() {
        let name = entry.file_name();
        let text = std::fs::read_to_string(entry.path()).expect("read");
        let text = if name == "ontology.yaml" {
            text.replace(
                "  metadata: schema/types.rs (the contract schema, not Σ)\n",
                "",
            )
        } else {
            text
        };
        std::fs::write(dst.join(name), text).expect("write");
    }
    let r = pv_in(
        scratch.path(),
        &["lint", "contracts", "--gate", "sigma", "--format", "json"],
    );
    assert_eq!(r.code, 3, "a Σ key nothing claims is exit 3\n{}", show(&r));
    assert!(
        r.stderr.contains("metadata") && r.stderr.contains("no reader"),
        "the refusal names the key\n{}",
        show(&r)
    );
}

#[test]
fn a_conforming_study_contract_passes_with_its_own_properties_as_focus_predicates() {
    let r = gate("entity-props-ok", "shapes");
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["focus_nodes_n"], 1, "{}", show(&r));
    assert_eq!(v["violations"], 0, "{}", show(&r));
}

#[test]
fn a_value_outside_the_in_list_fails_naming_the_property_and_the_value() {
    // apex EV-21 FALSIFY-ONT-021.
    let r = gate("entity-props-bad-value", "shapes");
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Fail", "{}", show(&r));
    let msgs = messages(&v);
    assert!(
        msgs.iter().any(|m| m.contains("ont:study/scale")
            && m.contains("quadratic")
            && m.contains("is not one of")),
        "the violation names the property and the value: {msgs:?}"
    );
}

#[test]
fn a_property_the_shape_does_not_declare_fails_on_closed_naming_it() {
    // apex EV-21 FALSIFY-ONT-021b.
    let r = gate("entity-props-undeclared-key", "shapes");
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    let msgs = messages(&v);
    assert!(
        msgs.iter()
            .any(|m| m.contains("(closed)") && m.contains("ont:study/subgroup")),
        "the violation names the undeclared property: {msgs:?}"
    );
}

#[test]
fn the_properties_are_namespaced_by_the_entity_type_and_land_in_the_extraction() {
    let dir = fixture("entity-props-ok");
    let r = pv_in(&dir, &["extract", "contracts"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let nt = std::fs::read_to_string(dir.join("contracts/contracts.nt")).expect("contracts.nt");
    assert!(nt.contains("/study/scale> \"linear\""), "{nt}");
    assert!(nt.contains("/study/vintage> \"2026-09-12\""), "{nt}");
    // Never under ont:, which would let a shape over one entity type constrain another's `scale`.
    assert!(!nt.contains("/v1alpha1/scale>"), "{nt}");
    let _ = std::fs::remove_file(dir.join("contracts/contracts.nt"));
    let _ = std::fs::remove_file(dir.join("contracts/shapes.ttl"));
}
