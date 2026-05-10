//! Integration test for `contracts/apr-rerank-v1.yaml`.
//!
//! HELIX-IDEA-006 Phase 1. Loader/validator that promotes the
//! reranking contract from DRAFT to ACTIVE for the pure-math
//! subset. Same pattern as `apr_mcp_server_contract.rs`. Asserts:
//!
//! 1. The YAML file exists and parses as valid YAML.
//! 2. Top-level `status: ACTIVE`.
//! 3. **Exactly 2** entries in `falsification_conditions`:
//!    FALSIFY-RERANK-RRF-002 (input-order invariance) and
//!    FALSIFY-RERANK-MMR-002 (λ=1 identity). Phases 2+ will amend
//!    the contract to add the remaining gates from §2.6.
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
        .join("apr-rerank-v1.yaml")
}

fn load_contract() -> ContractRoot {
    let path = contract_path();
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_yaml::from_str(&yaml)
        .unwrap_or_else(|e| panic!("parse {} as apr-rerank-v1 contract: {e}", path.display()))
}

#[test]
fn apr_rerank_contract_parses() {
    let _ = load_contract();
}

#[test]
fn apr_rerank_contract_is_active() {
    let contract = load_contract();
    assert_eq!(
        contract.status, "ACTIVE",
        "apr-rerank-v1.yaml must be ACTIVE for Phase 1 (RRF symmetry + MMR λ=1 shipped)."
    );
}

#[test]
fn apr_rerank_contract_has_exactly_two_phase_one_conditions() {
    let contract = load_contract();
    assert_eq!(
        contract.falsification_conditions.len(),
        2,
        "Phase 1 ships exactly 2 falsification gates \
         (FALSIFY-RERANK-RRF-002 + FALSIFY-RERANK-MMR-002); contract has {}. \
         Phase 2+ amendments must update both the YAML and this test in the same PR.",
        contract.falsification_conditions.len()
    );
}

#[test]
fn apr_rerank_contract_ids_are_rrf_002_and_mmr_002() {
    let contract = load_contract();
    let actual: Vec<String> = contract
        .falsification_conditions
        .iter()
        .map(|c| c.id.clone())
        .collect();
    assert_eq!(
        actual,
        vec![
            "FALSIFY-RERANK-RRF-002".to_string(),
            "FALSIFY-RERANK-MMR-002".to_string(),
        ],
    );
}

#[test]
fn apr_rerank_contract_every_test_file_exists() {
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
fn apr_rerank_contract_every_condition_is_enforced() {
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
