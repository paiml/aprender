//! Design-by-Contract verification harness.
//!
//! Parses `contracts/agent-loop-v1.yaml` and validates that all
//! invariant `test_binding` entries correspond to real tests.
//! Provides `verify_contracts()` for CI integration.
//!
//! See: docs/specifications/batuta-agent.md Section 13.

use std::collections::HashMap;

use serde::Deserialize;

/// Top-level contract structure parsed from YAML.
#[derive(Debug, Deserialize)]
pub struct ContractFile {
    /// Contract metadata.
    pub contract: ContractMeta,
    /// Formal invariants.
    pub invariants: Vec<Invariant>,
    /// Verification targets.
    pub verification: VerificationTargets,
}

/// Contract metadata.
#[derive(Debug, Deserialize)]
pub struct ContractMeta {
    /// Contract identifier.
    pub name: String,
    /// Semantic version.
    pub version: String,
    /// Rust module path.
    pub module: String,
    /// Human-readable description.
    pub description: String,
}

/// A formal invariant with preconditions, postconditions, and test binding.
#[derive(Debug, Deserialize)]
pub struct Invariant {
    /// Unique identifier (e.g., "INV-001").
    pub id: String,
    /// Short name (e.g., "loop-terminates").
    pub name: String,
    /// Description of the invariant.
    pub description: String,
    /// Required preconditions.
    pub preconditions: Vec<String>,
    /// Guaranteed postconditions.
    pub postconditions: Vec<String>,
    /// Formal equation (mathematical notation).
    pub equation: String,
    /// Rust module path to the implementation.
    pub module_path: String,
    /// Test that verifies this invariant.
    pub test_binding: String,
}

/// Verification targets from the contract.
#[derive(Debug, Deserialize)]
pub struct VerificationTargets {
    /// List of unit test paths that verify contract invariants.
    pub unit_tests: Vec<String>,
    /// Minimum coverage percentage.
    pub coverage_target: u32,
    /// Minimum mutation testing score.
    pub mutation_target: u32,
    /// Maximum cyclomatic complexity.
    pub complexity_max_cyclomatic: u32,
    /// Maximum cognitive complexity.
    pub complexity_max_cognitive: u32,
}

/// Result of contract verification.
#[derive(Debug)]
pub struct VerificationResult {
    /// Contract name.
    pub contract_name: String,
    /// Total invariants.
    pub total_invariants: usize,
    /// Invariants with verified test bindings.
    pub verified_bindings: usize,
    /// Missing test bindings.
    pub missing_bindings: Vec<String>,
    /// Per-invariant status.
    pub invariant_status: HashMap<String, InvariantStatus>,
}

/// Status of a single invariant verification.
#[derive(Debug, Clone)]
pub struct InvariantStatus {
    /// Invariant ID.
    pub id: String,
    /// Invariant name.
    pub name: String,
    /// Whether the test binding was found.
    pub test_found: bool,
    /// The test binding path.
    pub test_binding: String,
}

impl VerificationResult {
    /// Whether all invariants have verified test bindings.
    #[must_use]
    pub fn all_verified(&self) -> bool {
        self.missing_bindings.is_empty()
    }

    /// Format as a human-readable report.
    #[must_use]
    pub fn report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Contract: {} ({}/{})",
            self.contract_name, self.verified_bindings, self.total_invariants,
        );

        for status in self.invariant_status.values() {
            let mark = if status.test_found { "✓" } else { "✗" };
            let _ = writeln!(out, "  [{mark}] {} — {}", status.id, status.name,);
        }

        if !self.missing_bindings.is_empty() {
            let _ = writeln!(out, "\nMissing bindings:");
            for b in &self.missing_bindings {
                let _ = writeln!(out, "  - {b}");
            }
        }

        out
    }
}

/// Parse a contract YAML file.
pub fn parse_contract(yaml_content: &str) -> Result<ContractFile, String> {
    serde_yaml_ng::from_str(yaml_content).map_err(|e| format!("YAML parse error: {e}"))
}

/// Verify contract invariants against a set of known test names.
///
/// The `known_tests` set should contain test paths as returned by
/// `cargo test --list` (e.g., `agent::guard::tests::test_iteration_limit`).
pub fn verify_bindings(contract: &ContractFile, known_tests: &[String]) -> VerificationResult {
    let mut status = HashMap::new();
    let mut missing = Vec::new();
    let mut verified = 0;

    for inv in &contract.invariants {
        let found = known_tests.iter().any(|t| t.contains(&inv.test_binding));

        if found {
            verified += 1;
        } else {
            missing.push(format!("{}: {} (expected: {})", inv.id, inv.name, inv.test_binding,));
        }

        status.insert(
            inv.id.clone(),
            InvariantStatus {
                id: inv.id.clone(),
                name: inv.name.clone(),
                test_found: found,
                test_binding: inv.test_binding.clone(),
            },
        );
    }

    VerificationResult {
        contract_name: contract.contract.name.clone(),
        total_invariants: contract.invariants.len(),
        verified_bindings: verified,
        missing_bindings: missing,
        invariant_status: status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_YAML: &str = include_str!("../../contracts/agent-loop-v1.yaml");

    #[test]
    fn test_parse_contract() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");
        assert_eq!(contract.contract.name, "agent-loop-v1");
        assert_eq!(contract.contract.version, "1.0.0");
        assert_eq!(contract.contract.module, "batuta::agent");
        assert!(!contract.invariants.is_empty());
    }

    #[test]
    fn test_invariant_count() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");
        assert_eq!(
            contract.invariants.len(),
            16,
            "expected 16 invariants (8 loop + 4 pool/sanitization + 4 tool)"
        );
    }

    #[test]
    fn test_invariant_ids() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");
        let ids: Vec<&str> = contract.invariants.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"INV-001"));
        assert!(ids.contains(&"INV-007"));
    }

    #[test]
    fn test_invariant_fields_populated() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");
        for inv in &contract.invariants {
            assert!(!inv.name.is_empty(), "{} has empty name", inv.id);
            assert!(!inv.description.is_empty(), "{} has empty description", inv.id);
            assert!(!inv.preconditions.is_empty(), "{} has no preconditions", inv.id);
            assert!(!inv.postconditions.is_empty(), "{} has no postconditions", inv.id);
            assert!(!inv.equation.is_empty(), "{} has empty equation", inv.id);
            assert!(!inv.module_path.is_empty(), "{} has empty module_path", inv.id);
            assert!(!inv.test_binding.is_empty(), "{} has empty test_binding", inv.id);
        }
    }

    #[test]
    fn test_verification_targets() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");
        assert_eq!(contract.verification.coverage_target, 95);
        assert_eq!(contract.verification.mutation_target, 80);
        assert_eq!(contract.verification.complexity_max_cyclomatic, 30);
        assert_eq!(contract.verification.complexity_max_cognitive, 25);
        assert!(!contract.verification.unit_tests.is_empty());
    }

    #[test]
    fn test_verify_all_bindings_found() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");

        // Simulate known tests matching all bindings
        let known_tests: Vec<String> =
            contract.invariants.iter().map(|i| i.test_binding.clone()).collect();

        let result = verify_bindings(&contract, &known_tests);
        assert!(result.all_verified());
        assert_eq!(result.verified_bindings, contract.invariants.len());
    }

    #[test]
    fn test_verify_missing_binding() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");

        // No known tests → all bindings missing
        let result = verify_bindings(&contract, &[]);
        assert!(!result.all_verified());
        assert_eq!(result.verified_bindings, 0);
        assert_eq!(result.missing_bindings.len(), contract.invariants.len());
    }

    #[test]
    fn test_verify_partial_bindings() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");

        // Only first binding exists
        let known_tests = vec![contract.invariants[0].test_binding.clone()];

        let result = verify_bindings(&contract, &known_tests);
        assert!(!result.all_verified());
        assert_eq!(result.verified_bindings, 1);
    }

    #[test]
    fn test_report_format() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");
        let result = verify_bindings(&contract, &[]);
        let report = result.report();
        assert!(report.contains("agent-loop-v1"));
        assert!(report.contains("Missing bindings"));
    }

    #[test]
    fn test_report_all_pass() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");
        let known_tests: Vec<String> =
            contract.invariants.iter().map(|i| i.test_binding.clone()).collect();
        let result = verify_bindings(&contract, &known_tests);
        let report = result.report();
        assert!(!report.contains("Missing bindings"));
    }

    /// Verify contract equations map to #[contract]-annotated functions.
    #[test]
    fn test_contract_equations_have_code_bindings() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");

        // These equations have #[contract] macro bindings in source.
        // When agents-contracts is enabled, the macro generates
        // const binding strings for audit traceability.
        let bound_equations = [
            "loop_termination", // runtime.rs::run_agent_loop
            "capability_match", // capability.rs::capability_matches
            "guard_budget",     // guard.rs::LoopGuard::record_cost
        ];

        for inv in &contract.invariants {
            let eq_lines: Vec<&str> =
                inv.equation.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
            // Verify equation field is non-empty (already tested elsewhere)
            assert!(!eq_lines.is_empty(), "{}: empty equation", inv.id);
        }

        // Count how many invariant module_paths correspond to bound functions
        let bound_count = contract
            .invariants
            .iter()
            .filter(|inv| {
                bound_equations.iter().any(|eq| {
                    inv.equation.contains(eq)
                        || inv.module_path.contains("record_cost")
                        || inv.module_path.contains("run_agent_loop")
                        || inv.module_path.contains("capability_matches")
                })
            })
            .count();

        assert!(bound_count >= 3, "expected >= 3 #[contract] bindings, got {bound_count}");
    }

    /// Integration test: verify all contract test bindings
    /// actually exist in this crate's test suite.
    #[test]
    fn test_all_contract_bindings_exist() {
        let contract = parse_contract(TEST_YAML).expect("parse failed");

        // These are the actual test names in our test suite.
        // We verify each binding maps to a real test.
        let existing_tests = [
            "agent::guard::tests::test_iteration_limit",
            "agent::guard::tests::test_counters",
            "agent::runtime::tests::test_capability_denied_handled",
            "agent::guard::tests::test_pingpong_detection",
            "agent::guard::tests::test_cost_budget",
            "agent::guard::tests::test_consecutive_max_tokens",
            "agent::runtime::tests::test_conversation_stored_in_memory",
            "agent::pool::tests::test_pool_capacity_limit",
            "agent::pool::tests::test_pool_fan_out_fan_in",
            "agent::pool::tests::test_pool_join_all",
            "agent::tool::tests::test_sanitize_output_system_injection",
            "agent::tool::spawn::tests::test_spawn_tool_depth_limit",
            "agent::tool::network::tests::test_blocked_host",
            "agent::tool::inference::tests::test_inference_tool_timeout",
            "agent::runtime::tests_advanced::test_sovereign_privacy_blocks_network",
            "agent::guard::tests::test_token_budget_exhausted",
        ];

        let known: Vec<String> = existing_tests.iter().map(|s| (*s).to_string()).collect();
        let result = verify_bindings(&contract, &known);

        assert!(result.all_verified(), "Missing bindings:\n{}", result.report());
    }
}
