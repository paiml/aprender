//! Generate Rust trait + test scaffolding from a contract.
//!
//! Usage:
//!   cargo run --example scaffold -- contracts/softmax-kernel-v1.yaml

use std::path::PathBuf;
use std::process;

use provable_contracts::scaffold::{generate_contract_tests, generate_trait};
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: scaffold <contract.yaml>");
            process::exit(1);
        },
        PathBuf::from,
    );

    let contract = parse_contract(&path).unwrap_or_else(|e| {
        eprintln!("Failed to parse {}: {e}", path.display());
        process::exit(1);
    });

    println!("// === Trait Definition ===\n");
    println!("{}", generate_trait(&contract));
    println!("// === Contract Tests ===\n");
    println!("{}", generate_contract_tests(&contract));
}
