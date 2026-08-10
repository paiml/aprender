//! Integration test for `contracts/apr-mcp-server-v1.yaml`.
//!
//! This is the loader/validator that promotes the MCP server contract from
//! DRAFT to ACTIVE. It asserts:
//!
//! 1. The YAML file exists and parses as valid YAML.
//! 2. Top-level `status: ACTIVE`.
//! 3. Exactly 13 entries in `falsification_conditions`, with ids
//!    FALSIFY-MCP-001 through FALSIFY-MCP-013 (no gaps, no duplicates).
//! 4. Every entry has a non-empty `test_file` that exists on disk relative
//!    to the workspace root, plus a non-empty `test_name` and
//!    `status: ENFORCED`.
//! 5. The named `test_name` actually appears in the named `test_file`.
//! 6. `description` and `id` are present on every entry.
//!
//! Failing any of these means the contract asserts something that does not
//! ship. A renamed or deleted test fails this test loudly before the MCP
//! crate tests even get a chance to compile.
//!
//! Check 5 is new. Until now the loader only asserted the FILE existed, so
//! renaming a test inside a file that still exists left the contract naming
//! a function nobody runs — which is exactly what happened to
//! FALSIFY-MCP-007 when its assertion was inverted to match the MCP
//! lifecycle. File-exists is shape; name-resolves is behaviour.

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
        .join("apr-mcp-server-v1.yaml")
}

fn load_contract() -> ContractRoot {
    let path = contract_path();
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        panic!(
            "parse {} as apr-mcp-server-v1 contract: {e}",
            path.display()
        )
    })
}

#[test]
fn apr_mcp_server_contract_parses() {
    // Just loading is the assertion — serde_yaml + our ContractRoot shape
    // fails if any required field is missing or mistyped.
    let _ = load_contract();
}

#[test]
fn apr_mcp_server_contract_is_active() {
    let contract = load_contract();
    assert_eq!(
        contract.status, "ACTIVE",
        "apr-mcp-server-v1.yaml must be promoted to ACTIVE (M4 milestone)"
    );
}

#[test]
fn apr_mcp_server_contract_has_exactly_thirteen_conditions() {
    let contract = load_contract();
    assert_eq!(
        contract.falsification_conditions.len(),
        13,
        "spec defines 13 FALSIFY-MCP gates; contract has {}",
        contract.falsification_conditions.len()
    );
}

#[test]
fn apr_mcp_server_contract_ids_are_falsify_mcp_001_through_013() {
    let contract = load_contract();
    let actual: Vec<String> = contract
        .falsification_conditions
        .iter()
        .map(|c| c.id.clone())
        .collect();
    let expected: Vec<String> = (1..=13).map(|n| format!("FALSIFY-MCP-{n:03}")).collect();
    assert_eq!(
        actual, expected,
        "ids must be exactly FALSIFY-MCP-001..013 in order (no gaps, no duplicates)"
    );
}

/// The contract must name a test that RESOLVES, not merely a file that
/// exists. A rename inside a surviving file used to slip through silently.
#[test]
fn apr_mcp_server_contract_every_test_name_resolves_in_its_file() {
    let contract = load_contract();
    let root = workspace_root();
    for cond in &contract.falsification_conditions {
        let full = root.join(&cond.test_file);
        let source = std::fs::read_to_string(&full)
            .unwrap_or_else(|e| panic!("{}: read {}: {e}", cond.id, full.display()));
        assert!(
            source.contains(&format!("fn {}(", cond.test_name)),
            "{}: contract names test `{}` but no `fn {}(` appears in {} — \
             a renamed test leaves the contract asserting something nobody runs",
            cond.id,
            cond.test_name,
            cond.test_name,
            cond.test_file
        );
    }
}

#[test]
fn apr_mcp_server_contract_every_test_file_exists() {
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
fn apr_mcp_server_contract_every_condition_is_enforced() {
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
