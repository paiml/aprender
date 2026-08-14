//! Integration test for `contracts/apr-serve-api-key-auth-v1.yaml`.
//!
//! HELIX-IDEA-009. Same pattern as `apr_mcp_server_contract.rs`: this test
//! is the loader/validator that promotes the auth contract from DRAFT to
//! ACTIVE. It asserts:
//!
//! 1. The YAML file exists and parses as valid YAML.
//! 2. Top-level `status: ACTIVE`.
//! 3. Exactly 3 entries in `falsification_conditions`, with ids
//!    FALSIFY-AUTH-001, FALSIFY-AUTH-002, FALSIFY-AUTH-003 (no gaps,
//!    no duplicates).
//! 4. Every entry has a non-empty `test_file` that exists on disk
//!    relative to the workspace root, plus a non-empty `test_name` and
//!    `status: ENFORCED`.
//! 5. `description` and `id` are present on every entry.
//!
//! Renaming or deleting any FALSIFY-AUTH test fails this test loudly
//! before apr-cli compiles.

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
        .join("apr-serve-api-key-auth-v1.yaml")
}

fn load_contract() -> ContractRoot {
    let path = contract_path();
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        panic!(
            "parse {} as apr-serve-api-key-auth-v1 contract: {e}",
            path.display()
        )
    })
}

#[test]
fn apr_serve_api_key_auth_contract_parses() {
    let _ = load_contract();
}

#[test]
fn apr_serve_api_key_auth_contract_is_active() {
    let contract = load_contract();
    assert_eq!(
        contract.status, "ACTIVE",
        "apr-serve-api-key-auth-v1.yaml must be ACTIVE — HELIX-IDEA-009 ships it on first land."
    );
}

#[test]
fn apr_serve_api_key_auth_contract_has_exactly_three_conditions() {
    let contract = load_contract();
    assert_eq!(
        contract.falsification_conditions.len(),
        3,
        "spec lists 3 FALSIFY-AUTH gates; contract has {}",
        contract.falsification_conditions.len()
    );
}

#[test]
fn apr_serve_api_key_auth_contract_ids_are_falsify_auth_001_through_003() {
    let contract = load_contract();
    let actual: Vec<String> = contract
        .falsification_conditions
        .iter()
        .map(|c| c.id.clone())
        .collect();
    let expected: Vec<String> = (1..=3).map(|n| format!("FALSIFY-AUTH-{n:03}")).collect();
    assert_eq!(
        actual, expected,
        "ids must be exactly FALSIFY-AUTH-001..003 in order (no gaps, no duplicates)"
    );
}

#[test]
fn apr_serve_api_key_auth_contract_every_test_file_exists() {
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

/// Every `test_name` must actually be a fn in its `test_file`.
///
/// #2465: this file's own header claimed "Renaming or deleting any FALSIFY-AUTH
/// test fails this test loudly before apr-cli compiles." It did not. The only
/// assertion on `test_name` was `!is_empty()`, so a name that matched nothing
/// passed — and two of the three did:
///
///   FALSIFY-AUTH-002  valid_bearer_passes_and_hash_path_is_constant_time -> absent
///   FALSIFY-AUTH-003  auth_module_uses_subtle_constanttimeeq             -> absent
///
/// Both had been renamed in the test files without the contract following.
/// Checking that the FILE exists is not checking that the TEST exists.
#[test]
fn apr_serve_api_key_auth_contract_every_test_name_exists_in_its_file() {
    let contract = load_contract();
    let root = workspace_root();
    for cond in &contract.falsification_conditions {
        let full = root.join(&cond.test_file);
        let src = std::fs::read_to_string(&full)
            .unwrap_or_else(|e| panic!("{}: read {}: {e}", cond.id, full.display()));
        let needle = format!("fn {}(", cond.test_name);
        assert!(
            src.contains(&needle),
            "{}: test_name `{}` is not defined in {} — the contract cites a test that \
             does not exist, so nothing enforces this gate.",
            cond.id,
            cond.test_name,
            cond.test_file
        );
    }
}

#[test]
fn apr_serve_api_key_auth_contract_every_condition_is_enforced() {
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
