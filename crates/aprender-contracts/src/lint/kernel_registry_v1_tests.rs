//! #4539 FALSIFY-KREG-001..003 — `contracts/kernel-registry-v1.yaml` against its own case table.
//!
//! The REAL contract and the REAL registry are staged, never fixture copies of them, so the table
//! cannot drift from the shape the gate runs. The pass arm is the committed registry; every other
//! row plants one defect in its first row and requires a refusal that names the focus node and
//! the property — a red that does not say which row is a red nobody can act on.

use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const CONTRACT: &str = "contracts/kernel-registry-v1.yaml";
const REF: &str = "crates/aprender-serve/kernel-registry.json";

fn real_registry() -> serde_json::Value {
    let text = std::fs::read_to_string(repo_root().join(REF)).expect("the committed registry");
    serde_json::from_str(&text).expect("the registry is JSON")
}

/// Stage the real contract plus `doc` at its `entity.ref`, and run the shapes gate on it.
fn gate(doc: &serde_json::Value) -> ShapesOutcome {
    let tmp = tempfile::tempdir().expect("tempdir");
    let contracts = tmp.path().join("contracts");
    std::fs::create_dir_all(&contracts).expect("contracts dir");
    std::fs::copy(
        repo_root().join(CONTRACT),
        contracts.join("kernel-registry-v1.yaml"),
    )
    .expect("stage the real contract");
    let data = tmp.path().join(REF);
    std::fs::create_dir_all(data.parent().expect("ref has a parent")).expect("ref dir");
    std::fs::write(&data, doc.to_string()).expect("stage the registry");
    run_shapes_gate(&contracts)
}

fn assert_refused(case: &str, doc: &serde_json::Value, property: &str) {
    match gate(doc) {
        ShapesOutcome::Ran { result, findings } => {
            assert!(!result.passed, "{case} was not refused");
            let all: Vec<&str> = findings.iter().map(|f| f.message.as_str()).collect();
            assert!(
                all.iter()
                    .any(|m| m.contains("kernel-registry-v1.kernels.0")
                        && m.contains(&format!("ont:kreg/{property}"))),
                "{case}: no finding names row 0 and `{property}`: {all:?}"
            );
        }
        other => panic!("{case}: expected Ran, got {other:?}"),
    }
}

fn with_row0(key: &str, value: Option<serde_json::Value>) -> serde_json::Value {
    let mut d = real_registry();
    let row = d["kernels"][0].as_object_mut().expect("row 0 is an object");
    match value {
        Some(v) => {
            row.insert(key.to_string(), v);
        }
        None => {
            row.remove(key);
        }
    }
    d
}

#[test]
fn kernel_registry_v1_the_real_registry_conforms_with_every_row_a_focus_node() {
    let doc = real_registry();
    let rows = doc["kernels"].as_array().expect("kernels").len();
    assert!(rows > 0);
    match gate(&doc) {
        ShapesOutcome::Ran { result, findings } => {
            assert!(result.passed, "{findings:?}");
            match &result.extra {
                Some(GateExtra::Shapes { by_shape, .. }) => {
                    let want = format!("kernel-registry-v1={rows}");
                    assert!(by_shape.contains(&want), "{want} not in {by_shape:?}");
                }
                other => panic!("{other:?}"),
            }
        }
        other => panic!("expected Ran, got {other:?}"),
    }
}

#[test]
fn falsify_kreg_001_a_col_major_row_is_refused() {
    let d = with_row0("layout", Some(serde_json::json!("col_major")));
    assert_refused("col_major", &d, "layout");
}

#[test]
fn falsify_kreg_002_a_row_with_no_qtype_is_refused() {
    assert_refused("no qtype", &with_row0("qtype", None), "qtype");
}

#[test]
fn kernel_registry_v1_a_qtype_outside_the_closed_set_is_refused() {
    let d = with_row0("qtype", Some(serde_json::json!("Q4K_RAW")));
    assert_refused("unknown qtype", &d, "qtype");
}

#[test]
fn falsify_kreg_003_a_tolerance_with_no_receipt_is_refused() {
    let d = with_row0("tolerance", Some(serde_json::json!("1e-3")));
    assert_refused("typed tolerance", &d, "tolerance");
}

#[test]
fn falsify_kreg_010_a_row_with_no_error_model_is_refused() {
    assert_refused(
        "no error_model",
        &with_row0("error_model", None),
        "error_model",
    );
}

#[test]
fn falsify_kreg_010_an_unknown_error_model_is_refused() {
    let d = with_row0("error_model", Some(serde_json::json!("EM-GUESS")));
    assert_refused("unknown error_model", &d, "error_model");
}

#[test]
fn falsify_kreg_010_an_undeclared_determinism_is_refused() {
    let d = with_row0("determinism", Some(serde_json::json!("mostly")));
    assert_refused("bad determinism", &d, "determinism");
    assert_refused(
        "no determinism",
        &with_row0("determinism", None),
        "determinism",
    );
}

#[test]
fn kernel_registry_v1_a_lowercase_or_spaced_requires_is_refused() {
    let d = with_row0("requires", Some(serde_json::json!("SHADER_F16 + SUBGROUP")));
    assert_refused("spaced requires", &d, "requires");
    assert_refused("no requires", &with_row0("requires", None), "requires");
}
