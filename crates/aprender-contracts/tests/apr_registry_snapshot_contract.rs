//! Integration test for `contracts/apr-registry-snapshot-v1.yaml`.
//!
//! HELIX-IDEA-007. Loader/validator that promotes the snapshot contract
//! from DRAFT to ACTIVE. Same pattern as `apr_mcp_server_contract.rs`:
//!
//! 1. The YAML file exists and parses as valid YAML.
//! 2. Top-level `status: ACTIVE`.
//! 3. Exactly 3 entries in `falsification_conditions`, with ids
//!    FALSIFY-SNAPSHOT-001..003 (no gaps, no duplicates).
//! 4. Every entry has a non-empty `test_file` that exists on disk
//!    relative to the workspace root, plus a non-empty `test_name` and
//!    `status: ENFORCED`.
//! 5. `description` and `id` are present on every entry.

use std::path::{Path, PathBuf};

#[derive(Debug, serde::Deserialize)]
struct ContractRoot {
    status: String,
    falsification_conditions: Vec<FalsificationCondition>,
}

#[derive(Debug, serde::Deserialize)]
struct FalsificationCondition {
    id: String,
    description: String,
    test_file: String,
    test_name: String,
    status: String,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root must resolve")
}

fn contract_path() -> PathBuf {
    workspace_root()
        .join("contracts")
        .join("apr-registry-snapshot-v1.yaml")
}

fn load_contract() -> ContractRoot {
    let path = contract_path();
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        panic!(
            "parse {} as apr-registry-snapshot-v1 contract: {e}",
            path.display()
        )
    })
}

#[test]
fn apr_registry_snapshot_contract_parses() {
    let _ = load_contract();
}

#[test]
fn apr_registry_snapshot_contract_is_active() {
    let contract = load_contract();
    assert_eq!(
        contract.status, "ACTIVE",
        "apr-registry-snapshot-v1.yaml must be ACTIVE — HELIX-IDEA-007 ships it on first land."
    );
}

#[test]
fn apr_registry_snapshot_contract_has_exactly_three_conditions() {
    let contract = load_contract();
    assert_eq!(
        contract.falsification_conditions.len(),
        3,
        "spec lists 3 FALSIFY-SNAPSHOT gates; contract has {}",
        contract.falsification_conditions.len()
    );
}

#[test]
fn apr_registry_snapshot_contract_ids_are_falsify_snapshot_001_through_003() {
    let contract = load_contract();
    let actual: Vec<String> = contract
        .falsification_conditions
        .iter()
        .map(|c| c.id.clone())
        .collect();
    let expected: Vec<String> = (1..=3)
        .map(|n| format!("FALSIFY-SNAPSHOT-{n:03}"))
        .collect();
    assert_eq!(
        actual, expected,
        "ids must be exactly FALSIFY-SNAPSHOT-001..003 in order (no gaps, no duplicates)"
    );
}

#[test]
fn apr_registry_snapshot_contract_every_test_file_exists() {
    let contract = load_contract();
    let root = workspace_root();
    for cond in &contract.falsification_conditions {
        assert!(
            !cond.test_file.is_empty(),
            "{}: test_file must be non-empty",
            cond.id
        );
        let full = root.join(&cond.test_file);
        assert!(
            full.is_file(),
            "{}: test_file {} does not exist on disk (resolved: {})",
            cond.id,
            cond.test_file,
            full.display()
        );
    }
}

#[test]
fn apr_registry_snapshot_contract_every_condition_is_enforced() {
    let contract = load_contract();
    for cond in &contract.falsification_conditions {
        assert_eq!(
            cond.status, "ENFORCED",
            "{}: status must be ENFORCED (got {:?})",
            cond.id, cond.status
        );
        assert!(
            !cond.test_name.is_empty(),
            "{}: test_name must be non-empty",
            cond.id
        );
        assert!(
            !cond.description.is_empty(),
            "{}: description must be non-empty",
            cond.id
        );
    }
}
