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

/// A scratch copy of a fixture's `contracts/`, for a command that writes beside its input (`pv extract` writes
/// `contracts.nt` and `shapes.ttl`): the tracked fixture is never written, and a failing assert leaves nothing behind
/// (quorum PMAT-4160, agy lanes 1 and 2).
fn scratch_copy(name: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("scratch");
    let to = dir.path().join("contracts");
    std::fs::create_dir_all(&to).expect("scratch contracts/");
    for e in std::fs::read_dir(fixture(name).join("contracts")).expect("fixture contracts/") {
        let e = e.expect("dir entry");
        std::fs::copy(e.path(), to.join(e.file_name())).expect("copy fixture file");
    }
    dir
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
    let scratch = scratch_copy("entity-props-ok");
    let dir = scratch.path();
    let r = pv_in(dir, &["extract", "contracts"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let nt = std::fs::read_to_string(dir.join("contracts/contracts.nt")).expect("contracts.nt");
    assert!(nt.contains("/study/scale> \"linear\""), "{nt}");
    assert!(nt.contains("/study/vintage> \"2026-09-12\""), "{nt}");
    // Never under ont:, which would let a shape over one entity type constrain another's `scale`.
    assert!(!nt.contains("/v1alpha1/scale>"), "{nt}");
}

// #4160 (apex EV-19b) — a shape can target ONE entity type. `ont:Contract` is every contract, so apex's closed
// study shape failed its 18 claim contracts and the claim shape failed the study. The pv-contract extractor now
// also types each contract `entity:<type>`, and `targetClass: entity:study` selects the study contracts only.
// DISCRIMINATION: `entity-types-scoped/` must PASS with exactly 3 focus nodes (a build without the entity class
// has 0; `ont:Contract` would give 4, the shape-only contract included); `entity-types-unscoped/` — the same two
// shapes on `ont:Contract` — must FAIL on the cross-firing, which is the gap measured; `entity-types-wrong-type/`
// adds a contract typed `study` that carries a claim's `row`, which the study shape must reject.

#[test]
fn two_closed_shapes_scoped_by_entity_type_pass_with_no_cross_firing() {
    let r = gate("entity-types-scoped", "shapes");
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["violations"], 0, "{}", show(&r));
    // study-shape-v1 (study) + PMAT-001, PMAT-002 (claim). claim-shape-v1 has no entity type: a focus of neither.
    assert_eq!(v["focus_nodes_n"], 3, "{}", show(&r));
}

#[test]
fn must_red_the_same_shapes_on_ont_contract_cross_fire() {
    let r = gate("entity-types-unscoped", "shapes");
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Fail", "{}", show(&r));
    let msgs = messages(&v);
    assert!(
        msgs.iter().any(|m| m.contains("contract/PMAT-001")
            && m.contains("study-shape-v1")
            && m.contains("(closed)")
            && m.contains("ont:claim/row")),
        "the study shape fires on a claim contract: {msgs:?}"
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("contract/study-shape-v1") && m.contains("claim-shape-v1")),
        "the claim shape fires on the study: {msgs:?}"
    );
}

#[test]
fn a_contract_of_the_wrong_type_fails_its_types_shape_and_only_that_one() {
    let r = gate("entity-types-wrong-type", "shapes");
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Fail", "{}", show(&r));
    let msgs = messages(&v);
    let fired: Vec<&String> = msgs.iter().filter(|m| m.contains("violates")).collect();
    assert!(
        fired.iter().any(|m| m.contains("contract/PMAT-003")
            && m.contains("(closed)")
            && m.contains("ont:study/row")),
        // properties are namespaced by the contract's OWN type, so a claim's `row` on a study is `study:row`
        "the study shape rejects a study-typed contract carrying a claim's row: {msgs:?}"
    );
    assert!(
        fired
            .iter()
            .all(|m| m.contains("contract/PMAT-003") && m.contains("study-shape-v1")),
        "nothing else fires — the claim contracts are still out of the study shape's scope: {msgs:?}"
    );
}

#[test]
fn the_entity_class_lands_in_the_extraction() {
    let scratch = scratch_copy("entity-types-scoped");
    let dir = scratch.path();
    let r = pv_in(dir, &["extract", "contracts"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let nt = std::fs::read_to_string(dir.join("contracts/contracts.nt")).expect("contracts.nt");
    let ty = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";
    assert!(
        nt.contains(&format!(
            "/contract/PMAT-001> {ty} <https://ont.paiml.dev/v1alpha1/entity/claim>"
        )),
        "{nt}"
    );
    assert!(
        nt.contains(&format!(
            "/contract/study-shape-v1> {ty} <https://ont.paiml.dev/v1alpha1/entity/study>"
        )),
        "{nt}"
    );
    assert!(
        !nt.contains("/contract/claim-shape-v1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://ont.paiml.dev/v1alpha1/entity/"),
        "no entity.type, no entity class: {nt}"
    );
}

/// PMAT-4160 (quorum round 3, sonnet seat 2): Σ refuses to DECLARE the reserved type, but a contract could still
/// CARRY it, and `pv extract` and `--gate shapes` never consult Σ. Both must refuse the extraction, with no graph
/// written, rather than emit an `entity/entity/<key>` predicate that aliases the class of type `<key>`.
#[test]
fn a_contract_carrying_the_reserved_entity_type_is_refused_by_extract_and_by_the_shapes_gate() {
    let scratch = scratch_copy("entity-types-scoped");
    let dir = scratch.path();
    let rogue = std::fs::read_to_string(dir.join("contracts/PMAT-001.yaml"))
        .expect("PMAT-001")
        .replace("name: PMAT-001", "name: ROGUE-001")
        .replace("  type: claim", "  type: entity")
        .replace("    row: EV-19b", "    study: EV-19b");
    assert!(
        rogue.contains("  type: entity"),
        "the fixture edit applied: {rogue}"
    );
    std::fs::write(dir.join("contracts/ROGUE-001.yaml"), rogue).expect("write rogue");

    let r = pv_in(dir, &["extract", "contracts"]);
    assert_ne!(r.code, 0, "{}", show(&r));
    assert!(
        r.stderr.contains("ROGUE-001") && r.stderr.contains("reserved"),
        "{}",
        show(&r)
    );
    assert!(
        !dir.join("contracts/contracts.nt").exists(),
        "a refused extraction writes no graph"
    );

    let r = pv_in(
        dir,
        &["lint", "contracts", "--gate", "shapes", "--format", "json"],
    );
    assert_ne!(r.code, 0, "{}", show(&r));
    assert!(
        format!("{}{}", r.stdout, r.stderr).contains("reserved"),
        "{}",
        show(&r)
    );

    // Control: the same corpus without the rogue contract extracts.
    std::fs::remove_file(dir.join("contracts/ROGUE-001.yaml")).expect("rm rogue");
    let r = pv_in(dir, &["extract", "contracts"]);
    assert_eq!(r.code, 0, "{}", show(&r));
}
