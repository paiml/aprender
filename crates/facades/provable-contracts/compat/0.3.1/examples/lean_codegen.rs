//! Generate Lean 4 theorem stubs from a single contract.
//!
//! Demonstrates Phase 7 codegen: contract YAML → Lean 4 source.
//!
//! Usage:
//!   cargo run --example `lean_codegen` -- contracts/softmax-kernel-v1.yaml

use std::path::PathBuf;
use std::process;

use provable_contracts::lean_gen::generate_lean_files;
use provable_contracts::schema::parse_contract;

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: lean_codegen <contract.yaml>");
            process::exit(1);
        },
        PathBuf::from,
    );

    let contract = parse_contract(&path).unwrap_or_else(|e| {
        eprintln!("Failed to parse {}: {e}", path.display());
        process::exit(1);
    });

    let files = generate_lean_files(&contract);

    if files.is_empty() {
        println!("No Lean metadata in {}.", path.display());
        println!("Add `lean:` blocks to proof obligations.");
        return;
    }

    println!(
        "Generated {} Lean files from {}:\n",
        files.len(),
        path.display()
    );

    for f in &files {
        println!("─── {} ───", f.path);
        println!("{}", f.content);
    }
}
