//! #4551 FALSIFY-RAWS-001 — `contracts/apr-raw-samples-v1.yaml` against its own case table.
//!
//! The REAL contract is staged, never a fixture copy of it, so the table cannot drift from the shape
//! the recorder runs. The pass arm is the committed real emit; every other row plants one defect a
//! plausible emitter could produce and requires the gate to refuse it.

use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const CONTRACT: &str = "contracts/apr-raw-samples-v1.yaml";
const REF: &str = "evidence/perf/raw-samples-v1/example-lambda-cpu.json";

fn real_emit() -> serde_json::Value {
    let text = std::fs::read_to_string(repo_root().join(REF)).expect("the committed real emit");
    serde_json::from_str(&text).expect("the real emit is JSON")
}

/// Stage the real contract plus `doc` at its `entity.ref`, and run the shapes gate on it.
fn gate(doc: &serde_json::Value) -> ShapesOutcome {
    let tmp = tempfile::tempdir().expect("tempdir");
    let contracts = tmp.path().join("contracts");
    std::fs::create_dir_all(&contracts).expect("contracts dir");
    std::fs::copy(
        repo_root().join(CONTRACT),
        contracts.join("apr-raw-samples-v1.yaml"),
    )
    .expect("stage the real contract");
    let data = tmp.path().join(REF);
    std::fs::create_dir_all(data.parent().expect("ref has a parent")).expect("ref dir");
    std::fs::write(&data, doc.to_string()).expect("stage the document");
    run_shapes_gate(&contracts)
}

fn assert_passes(doc: &serde_json::Value) {
    match gate(doc) {
        ShapesOutcome::Ran { result, findings } => assert!(result.passed, "{findings:?}"),
        other => panic!("expected Ran, got {other:?}"),
    }
}

fn assert_refused(case: &str, doc: &serde_json::Value) {
    match gate(doc) {
        ShapesOutcome::Ran { result, findings } => {
            assert!(!result.passed, "{case} was not refused");
            assert!(!findings.is_empty(), "{case}: a red with no finding");
        }
        other => panic!("{case}: expected Ran, got {other:?}"),
    }
}

#[test]
fn raw_samples_v1_the_real_emit_conforms() {
    assert_passes(&real_emit());
}

#[test]
fn raw_samples_v1_a_derived_statistic_is_refused() {
    let mut d = real_emit();
    d["tokens_per_second"] = serde_json::json!(12.5);
    assert_refused("root tokens_per_second", &d);

    let mut d = real_emit();
    d["samples"][0]["decode_tok_s"] = serde_json::json!(40.0);
    assert_refused("sample decode_tok_s", &d);
}

#[test]
fn raw_samples_v1_identity_it_could_not_read_is_refused() {
    for key in ["host", "binary_sha256", "model_sha256"] {
        let mut d = real_emit();
        d[key] = serde_json::Value::Null;
        assert_refused(&format!("null {key}"), &d);
    }
    let mut d = real_emit();
    d["binary_sha256"] = serde_json::json!("28d483d8");
    assert_refused("short binary digest", &d);
}

#[test]
fn raw_samples_v1_no_samples_and_foreign_values_are_refused() {
    let mut d = real_emit();
    d["samples"] = serde_json::json!([]);
    assert_refused("empty samples", &d);

    let mut d = real_emit();
    d["backend"] = serde_json::json!("gpu");
    assert_refused("undeclared backend", &d);

    let mut d = real_emit();
    d["schema"] = serde_json::json!("raw-samples-v2");
    assert_refused("foreign schema", &d);

    let mut d = real_emit();
    d["samples"][0]["completion_tokens"] = serde_json::json!("32");
    assert_refused("string token count", &d);
}
