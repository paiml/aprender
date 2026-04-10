use super::*;

// ── bootstrap_playbook_from_contract coverage (0% → full) ─────────────────

/// Minimal family YAML used across bootstrap tests.
const BOOTSTRAP_FAMILY_YAML: &str = r#"
family: testfam
display_name: "Test Family"
vendor: TestVendor
architectures:
  - TestForCausalLM
hf_pattern: "TestOrg/Test*"

size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 16
    num_heads: 8
    num_kv_heads: 2
    intermediate_dim: 2048
    vocab_size: 32000

constraints:
  attention_type: gqa
  activation: silu
  norm_type: rmsnorm
  has_bias: false
  tied_embeddings: false
  positional_encoding: rope
  mlp_type: swiglu

certification:
  size_categories:
    1b: small
"#;

/// Family YAML with no constraints section (to test that error path).
const NO_CONSTRAINTS_YAML: &str = r#"
family: nocon
display_name: "No Constraints"
vendor: TestVendor
architectures:
  - NoConForCausalLM
hf_pattern: "TestOrg/NoCon*"

size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 512
    num_layers: 4
    num_heads: 4
    num_kv_heads: 4
"#;

/// Write a family YAML to a temp directory and return the dir.
fn make_contracts_dir(filename: &str, content: &str) -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().expect("temp dir");
    std::fs::write(dir.path().join(filename), content).expect("write family yaml");
    dir
}

#[test]
fn test_bootstrap_missing_family_dir() {
    // Contracts path does not exist → family not found
    let result = bootstrap_playbook_from_contract(
        "testfam",
        "1b",
        "TestOrg/Test-1B-Instruct",
        "mvp",
        std::path::Path::new("/nonexistent/contracts"),
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("testfam"),
        "Error should mention family name, got: {err}"
    );
}

#[test]
fn test_bootstrap_family_file_missing() {
    // Contracts dir exists but family YAML is absent
    let dir = tempfile::TempDir::new().expect("temp dir");
    let result = bootstrap_playbook_from_contract(
        "testfam",
        "1b",
        "TestOrg/Test-1B-Instruct",
        "mvp",
        dir.path(),
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("testfam"));
}

#[test]
fn test_bootstrap_size_variant_not_found() {
    // Family YAML exists but the requested size is not in it
    let dir = make_contracts_dir("testfam.yaml", BOOTSTRAP_FAMILY_YAML);
    let result = bootstrap_playbook_from_contract(
        "testfam",
        "99b", // non-existent size
        "TestOrg/Test-99B-Instruct",
        "mvp",
        dir.path(),
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("99b"),
        "Error should mention missing size, got: {err}"
    );
    assert!(
        err.contains("1b"),
        "Error should list available sizes, got: {err}"
    );
}

#[test]
fn test_bootstrap_no_constraints_in_family() {
    // Family YAML exists with a valid size, but has no constraints block
    let dir = make_contracts_dir("nocon.yaml", NO_CONSTRAINTS_YAML);
    let result =
        bootstrap_playbook_from_contract("nocon", "1b", "TestOrg/NoCon-1B", "smoke", dir.path());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("constraints") || err.contains("nocon"),
        "Error should mention constraints, got: {err}"
    );
}

#[test]
fn test_bootstrap_success_produces_yaml() {
    // Happy path: family + size + constraints all present → valid YAML output
    let dir = make_contracts_dir("testfam.yaml", BOOTSTRAP_FAMILY_YAML);
    let result = bootstrap_playbook_from_contract(
        "testfam",
        "1b",
        "TestOrg/Test-1B-Instruct",
        "mvp",
        dir.path(),
    );
    assert!(result.is_ok(), "Expected Ok but got: {:?}", result.err());
    let yaml = result.unwrap();
    // The generated YAML must at minimum be non-empty and reference the model
    assert!(!yaml.is_empty());
    assert!(
        yaml.contains("TestOrg/Test-1B-Instruct") || yaml.contains("testfam"),
        "Generated YAML should reference the model or family, got: {yaml}"
    );
}

#[test]
fn test_bootstrap_smoke_tier() {
    let dir = make_contracts_dir("testfam.yaml", BOOTSTRAP_FAMILY_YAML);
    let result =
        bootstrap_playbook_from_contract("testfam", "1b", "TestOrg/Test-1B", "smoke", dir.path());
    assert!(result.is_ok(), "Expected Ok but got: {:?}", result.err());
}

#[test]
fn test_bootstrap_deep_tier() {
    let dir = make_contracts_dir("testfam.yaml", BOOTSTRAP_FAMILY_YAML);
    let result =
        bootstrap_playbook_from_contract("testfam", "1b", "TestOrg/Test-1B", "deep", dir.path());
    assert!(result.is_ok(), "Expected Ok but got: {:?}", result.err());
}
