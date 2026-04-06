//! `pv codegen` — generate Rust `debug_assert`!() from YAML contracts.

use provable_contracts::codegen;
use std::path::Path;

pub fn run(contract_dir: &Path, output: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let contracts = codegen::generate_all(contract_dir);

    if contracts.is_empty() {
        println!("No contracts with preconditions/postconditions/invariants found.");
        return Ok(());
    }

    let mut total_pre = 0;
    let mut total_post = 0;
    let mut total_inv = 0;
    let mut total_lean = 0;

    for c in &contracts {
        total_pre += c.precondition_count;
        total_post += c.postcondition_count;
        total_inv += c.invariant_count;
        total_lean += c.lean_theorem_count;
    }

    println!("pv codegen — contract → Rust assertions");
    println!("========================================\n");
    println!("Contracts:      {}", contracts.len());
    println!("Preconditions:  {total_pre}");
    println!("Postconditions: {total_post}");
    println!("Invariants:     {total_inv}");
    println!("Lean theorems:  {total_lean}");

    let out_path = output.unwrap_or(Path::new("src/generated_contracts.rs"));

    // Create parent directory if it doesn't exist
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    codegen::write_rust_module(&contracts, out_path)?;
    println!("\nGenerated: {}", out_path.display());

    Ok(())
}
