//! Demonstrates loading and validating YAML contract files.
//!
//! This example shows how to:
//! 1. Parse a YAML contract from a string using `parse_contract_str`
//! 2. Parse a YAML contract from a file using `parse_contract`
//! 3. Run validation to detect errors and warnings
//! 4. Check the provability invariant (proof obligations, falsification tests, kani harnesses)
//!
//! Run with:
//!   cargo run --example validate_contracts
//!
//! Or validate a specific file:
//!   cargo run --example validate_contracts -- contracts/softmax-kernel-v1.yaml

use std::path::{Path, PathBuf};

use provable_contracts::error::Severity;
use provable_contracts::schema::{parse_contract, parse_contract_str, validate_contract};

/// An inline contract demonstrating the minimal valid structure.
const INLINE_CONTRACT: &str = r#"
metadata:
  version: "1.0.0"
  created: "2026-04-03"
  author: "Example Author"
  description: "Softmax kernel — numerically stable exponential normalization"
  references:
    - "Bridle (1990) Training Stochastic Model Recognition Algorithms"
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))"
    domain: "x ∈ ℝ^n, n ≥ 1"
    codomain: "σ(x) ∈ (0,1)^n"
    invariants:
      - "Σ σ(x)_i = 1.0 (normalization)"
      - "σ(x)_i > 0 for all i (strict positivity)"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|Σ σ(x)_i - 1.0| < ε"
    tolerance: 1.0e-6
    applies_to: all
  - type: bound
    property: "All outputs strictly positive"
    formal: "σ(x)_i > 0 for all i"
falsification_tests:
  - id: FALSIFY-SM-001
    rule: "normalization"
    prediction: "sum(output) ≈ 1.0 for any finite input"
    if_fails: "Missing max-subtraction trick causes numerical overflow"
  - id: FALSIFY-SM-002
    rule: "positivity"
    prediction: "output > 0 for all elements"
    if_fails: "exp underflow to zero"
kani_harnesses:
  - id: KANI-SM-001
    obligation: SM-INV-001
    property: "Softmax normalization"
    bound: 16
    strategy: stub_float
  - id: KANI-SM-002
    obligation: SM-BND-001
    property: "Softmax positivity"
    bound: 8
    strategy: exhaustive
"#;

/// A deliberately invalid contract to demonstrate error reporting.
const INVALID_CONTRACT: &str = r#"
metadata:
  version: ""
  description: "Missing references and empty version"
  references: []
equations: {}
proof_obligations: []
falsification_tests: []
"#;

fn main() {
    println!("=== provable-contracts: Validation Example ===\n");

    // --- Part 1: Parse and validate an inline contract ---
    println!("--- Part 1: Validate a well-formed inline contract ---\n");
    validate_from_string(INLINE_CONTRACT, "inline-softmax");

    // --- Part 2: Parse and validate a deliberately invalid contract ---
    println!("\n--- Part 2: Validate an invalid contract (expect errors) ---\n");
    validate_from_string(INVALID_CONTRACT, "inline-invalid");

    // --- Part 3: Check the provability invariant directly ---
    println!("\n--- Part 3: Provability invariant check ---\n");
    let contract = parse_contract_str(INLINE_CONTRACT).expect("valid YAML");
    let violations = contract.provability_violations();
    if violations.is_empty() {
        println!(
            "  Contract passes provability invariant: has {} obligation(s), {} test(s), {} harness(es)",
            contract.proof_obligations.len(),
            contract.falsification_tests.len(),
            contract.kani_harnesses.len(),
        );
    } else {
        println!("  Provability violations:");
        for v in &violations {
            println!("    - {v}");
        }
    }

    // --- Part 4: Validate a file from disk (if provided as CLI arg, or scan contracts/) ---
    println!("\n--- Part 4: Validate contract file(s) from disk ---\n");
    if let Some(path_arg) = std::env::args().nth(1) {
        validate_file(&PathBuf::from(path_arg));
    } else {
        // Try to find the contracts directory relative to the workspace root
        let contracts_dir = Path::new("contracts");
        if contracts_dir.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(contracts_dir)
                .expect("readable contracts/")
                .filter_map(Result::ok)
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|ext| ext == "yaml")
                })
                .collect();
            entries.sort_by_key(std::fs::DirEntry::path);

            let mut total_errors = 0;
            let mut total_warnings = 0;
            let mut valid_count = 0;

            for entry in entries.iter().take(5) {
                let path = entry.path();
                let (errors, warnings) = validate_file(&path);
                total_errors += errors;
                total_warnings += warnings;
                if errors == 0 {
                    valid_count += 1;
                }
                println!();
            }

            println!("Summary: validated {} contracts", entries.len().min(5));
            println!(
                "  {valid_count} valid, {total_errors} total error(s), {total_warnings} total warning(s)"
            );
            if entries.len() > 5 {
                println!("  (showing first 5 of {})", entries.len());
            }
        } else {
            println!("  No CLI argument and no contracts/ directory found.");
            println!("  Run from the workspace root or pass a YAML path as argument.");
        }
    }
}

/// Parse and validate a contract from a YAML string.
fn validate_from_string(yaml: &str, label: &str) {
    match parse_contract_str(yaml) {
        Ok(contract) => {
            println!(
                "  Parsed: \"{}\" v{} ({label})",
                contract.metadata.description, contract.metadata.version
            );
            println!(
                "  Equations: {} | Obligations: {} | Tests: {} | Harnesses: {}",
                contract.equations.len(),
                contract.proof_obligations.len(),
                contract.falsification_tests.len(),
                contract.kani_harnesses.len(),
            );

            let violations = validate_contract(&contract);
            print_violations(&violations);
        }
        Err(e) => {
            println!("  Parse error ({label}): {e}");
        }
    }
}

/// Parse and validate a contract from a YAML file on disk.
/// Returns (error_count, warning_count).
fn validate_file(path: &Path) -> (usize, usize) {
    print!("  {}: ", path.display());
    match parse_contract(path) {
        Ok(contract) => {
            let violations = validate_contract(&contract);
            let errors = violations
                .iter()
                .filter(|v| v.severity == Severity::Error)
                .count();
            let warnings = violations
                .iter()
                .filter(|v| v.severity == Severity::Warning)
                .count();

            if violations.is_empty() {
                println!("VALID (v{})", contract.metadata.version);
            } else {
                println!("{} error(s), {} warning(s)", errors, warnings);
                for v in &violations {
                    let tag = match v.severity {
                        Severity::Error => "ERROR",
                        Severity::Warning => "WARN",
                        Severity::Info => "INFO",
                    };
                    println!("    [{tag}] {}: {}", v.rule, v.message);
                }
            }
            (errors, warnings)
        }
        Err(e) => {
            println!("PARSE ERROR: {e}");
            (1, 0)
        }
    }
}

/// Print a summary of validation violations.
fn print_violations(violations: &[provable_contracts::error::Violation]) {
    let errors = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count();
    let warnings = violations
        .iter()
        .filter(|v| v.severity == Severity::Warning)
        .count();

    if violations.is_empty() {
        println!("  Result: VALID (no violations)");
    } else {
        println!("  Violations ({errors} error(s), {warnings} warning(s)):");
        for v in violations {
            let tag = match v.severity {
                Severity::Error => "ERROR",
                Severity::Warning => "WARN",
                Severity::Info => "INFO",
            };
            let loc = v.location.as_deref().unwrap_or("?");
            println!("    [{tag}] {} at {loc}: {}", v.rule, v.message);
        }
    }
}
