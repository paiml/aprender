//! Integration test for `contracts/apr-hnsw-persistence-v1.yaml`.
//!
//! HELIX-IDEA-001 (FULL — Phases 1-4). Loader/validator that promotes
//! the HNSW persistence contract from DRAFT to ACTIVE for the entire
//! pre-authored gate set. Same pattern as
//! `apr_mcp_server_contract.rs`. Asserts:
//!
//! 1. The YAML file exists and parses as valid YAML.
//! 2. Top-level `status: ACTIVE`.
//! 3. **Exactly 4** entries in `falsification_conditions`:
//!    FALSIFY-HNSW-PERSIST-001 (round-trip identity),
//!    FALSIFY-HNSW-PERSIST-002 (crash safety),
//!    FALSIFY-HNSW-PERSIST-003 (recall threshold), and
//!    FALSIFY-HNSW-PERSIST-004 (cold-open latency budget). All four
//!    pre-authored gates from §2.1 are now discharged.
//! 4. Every entry's `test_file` exists on disk, `test_name` is
//!    non-empty, and `status: ENFORCED`.

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
        .join("apr-hnsw-persistence-v1.yaml")
}

fn load_contract() -> ContractRoot {
    let path = contract_path();
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        panic!(
            "parse {} as apr-hnsw-persistence-v1 contract: {e}",
            path.display()
        )
    })
}

#[test]
fn apr_hnsw_persistence_contract_parses() {
    let _ = load_contract();
}

#[test]
fn apr_hnsw_persistence_contract_is_active() {
    let contract = load_contract();
    assert_eq!(
        contract.status, "ACTIVE",
        "apr-hnsw-persistence-v1.yaml must be ACTIVE — all 4 pre-authored gates discharged."
    );
}

#[test]
fn apr_hnsw_persistence_contract_has_exactly_four_conditions() {
    let contract = load_contract();
    assert_eq!(
        contract.falsification_conditions.len(),
        4,
        "HELIX-IDEA-001 ships exactly 4 falsification gates \
         (FALSIFY-HNSW-PERSIST-001..004); contract has {}. \
         Any future amendment must update both the YAML and this \
         test in the same PR.",
        contract.falsification_conditions.len()
    );
}

#[test]
fn apr_hnsw_persistence_contract_ids_are_persist_001_through_004() {
    let contract = load_contract();
    let actual: Vec<String> = contract
        .falsification_conditions
        .iter()
        .map(|c| c.id.clone())
        .collect();
    assert_eq!(
        actual,
        vec![
            "FALSIFY-HNSW-PERSIST-001".to_string(),
            "FALSIFY-HNSW-PERSIST-002".to_string(),
            "FALSIFY-HNSW-PERSIST-003".to_string(),
            "FALSIFY-HNSW-PERSIST-004".to_string(),
        ],
    );
}

#[test]
fn apr_hnsw_persistence_contract_test_file_exists() {
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
fn apr_hnsw_persistence_contract_condition_is_enforced() {
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
