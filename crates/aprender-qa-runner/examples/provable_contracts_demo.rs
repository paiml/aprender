//! Provable Contracts Demo
//!
//! Demonstrates the `pv lint` integration for machine-checkable contract
//! validation. Shows how the QA framework's behavioral invariants (I-1..I-5),
//! gateway preconditions (G0-G4), and MQS scoring obligations are expressed
//! as provable contract YAMLs and validated via the three-gate pipeline
//! (validate → audit → score).
//!
//! Run with:
//! ```bash
//! cargo run --example provable_contracts_demo -p apr-qa-runner
//! ```

#![allow(clippy::expect_used)]

use std::path::Path;

const CONTRACT_FILES: [&str; 4] = [
    "apr-format-invariants-v1.yaml",
    "gateway-contract-v1.yaml",
    "mqs-scoring-v1.yaml",
    "garbage-oracle-v1.yaml",
];

fn show_inventory(contracts_dir: &Path, binding_path: &Path) {
    println!("--- Contract Inventory ---\n");

    for name in &CONTRACT_FILES {
        let exists = contracts_dir.join(name).exists();
        let status = if exists { "OK" } else { "MISSING" };
        println!("  [{status}] {name}");
    }

    let binding_exists = binding_path.exists();
    println!(
        "  [{}] binding.yaml",
        if binding_exists { "OK" } else { "MISSING" }
    );
    println!();
}

fn show_metadata(contracts_dir: &Path) {
    println!("--- Contract Metadata ---\n");

    for name in &CONTRACT_FILES {
        let path = contracts_dir.join(name);
        if !path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(&path).expect("read contract");
        let yaml: serde_yaml::Value = serde_yaml::from_str(&content).expect("parse YAML");

        let metadata = &yaml["metadata"];
        let version = metadata["version"].as_str().unwrap_or("?");
        let description = metadata["description"].as_str().unwrap_or("?");
        let registry = metadata["registry"].as_bool().unwrap_or(false);

        let obligations = yaml["proof_obligations"].as_sequence().map_or(0, Vec::len);
        let tests = yaml["falsification_tests"]
            .as_sequence()
            .map_or(0, Vec::len);

        println!("  {name}");
        println!("    version:     {version}");
        println!("    description: {description}");
        println!("    registry:    {registry}");
        println!("    obligations: {obligations}");
        println!("    tests:       {tests}");
        println!();
    }
}

fn show_bindings(binding_path: &Path) {
    println!("--- Binding Registry (Traceability) ---\n");

    if binding_path.exists() {
        let content = std::fs::read_to_string(binding_path).expect("read binding");
        let yaml: serde_yaml::Value = serde_yaml::from_str(&content).expect("parse YAML");

        if let Some(bindings) = yaml["bindings"].as_sequence() {
            for binding in bindings {
                let contract = binding["contract"].as_str().unwrap_or("?");
                let equation = binding["equation"].as_str().unwrap_or("?");
                let function = binding["function"].as_str().unwrap_or("?");
                let status = binding["status"].as_str().unwrap_or("?");
                println!("  {contract}::{equation}");
                println!("    -> {function} [{status}]");
            }
            println!("\n  Total bindings: {}", bindings.len());
        }
    } else {
        println!("  binding.yaml not found — run from project root");
    }
    println!();
}

fn show_embedded_contract() {
    println!("--- Embedded Format Contract (Runtime) ---\n");

    let embedded = aprender_qa_runner::load_format_contract().expect("load embedded contract");
    println!("  version:    {}", embedded.version);
    println!("  invariants: {}", embedded.invariants.len());
    for inv in &embedded.invariants {
        println!("    {}: {}", inv.id, inv.name);
    }
    println!("  dtype_bytes: {}", embedded.dtype_bytes.mappings.len());
    println!("  tolerances:  {}", embedded.tolerances.len());
    println!();
}

fn run_pv_lint() {
    println!("--- Quality Gate: pv lint ---\n");
    println!("  Run: pv lint contracts/ --min-score 0.40 --binding contracts/binding.yaml");
    println!("  Or:  make contract-lint");
    println!();

    let pv_result = std::process::Command::new("pv")
        .args([
            "lint",
            "contracts/",
            "--min-score",
            "0.40",
            "--binding",
            "contracts/binding.yaml",
        ])
        .output();

    match pv_result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                println!("  {line}");
            }
            if output.status.success() {
                println!("\n  Result: PASS");
            } else {
                println!("\n  Result: FAIL (exit code {})", output.status);
            }
        }
        Err(_) => {
            println!("  pv not installed — install with: cargo install provable-contracts-cli");
        }
    }
}

fn main() {
    println!("=== Provable Contracts Demo (Spec §18) ===\n");

    let contracts_dir = Path::new("contracts");
    let binding_path = contracts_dir.join("binding.yaml");

    show_inventory(contracts_dir, &binding_path);
    show_metadata(contracts_dir);
    show_bindings(&binding_path);
    show_embedded_contract();
    run_pv_lint();

    println!("\n--- Summary ---\n");
    println!("  Provable contracts express QA framework invariants as");
    println!("  machine-checkable YAML validated by `pv lint`.");
    println!();
    println!("  Static (contracts/):   4 contracts, 21 obligations, 20 tests, 14 bindings");
    println!("  Runtime (embedded):    5 invariants (I-1..I-5), enforced in apr-qa-runner");
    println!("  Quality gate:          make contract-lint (in make check chain)");
    println!("  CI:                    .github/workflows/ci.yml → contract-lint job");
    println!("  Spec:                  §18 Provable Contracts Integration (v1.6.0)");
}
