//! Generate all code artifacts from a contract: trait, tests, Kani harnesses, probar tests.
//!
//! Usage:
//!   cargo run --example codegen -- contracts/softmax-kernel-v1.yaml

use std::path::PathBuf;
use std::process;

use provable_contracts::kani_gen::generate_kani_harnesses;
use provable_contracts::probar_gen::generate_probar_tests;
use provable_contracts::scaffold::{generate_contract_tests, generate_trait};
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: codegen <contract.yaml>");
            process::exit(1);
        },
        PathBuf::from,
    );

    let contract = parse_contract(&path).unwrap_or_else(|e| {
        eprintln!("Failed to parse {}: {e}", path.display());
        process::exit(1);
    });

    let name = contract
        .metadata
        .description
        .replace(' ', "_")
        .to_lowercase();
    let version = &contract.metadata.version;

    println!("// ============================================");
    println!("// Contract: {} v{version}", contract.metadata.description);
    println!("// Equations: {}", contract.equations.len());
    println!("// Obligations: {}", contract.proof_obligations.len());
    println!(
        "// Falsification tests: {}",
        contract.falsification_tests.len()
    );
    println!("// Kani harnesses: {}", contract.kani_harnesses.len());
    println!("// ============================================\n");

    // 1. Trait
    println!("// ---- {name}_trait.rs ----\n");
    println!("{}", generate_trait(&contract));

    // 2. Contract tests
    println!("// ---- {name}_contract_tests.rs ----\n");
    println!("{}", generate_contract_tests(&contract));

    // 3. Kani harnesses
    println!("// ---- {name}_kani.rs ----\n");
    println!("{}", generate_kani_harnesses(&contract));

    // 4. Probar property tests
    println!("// ---- {name}_probar.rs ----\n");
    println!("{}", generate_probar_tests(&contract));
}
