//! `pv verify-structure` — verify model architecture matches contracts.
//!
//! Two modes:
//! 1. Contract-only (no model file): enumerate expected tensors from
//!    apr-architecture-schema-v1 equations for a given config.
//! 2. With model file (future): compare expected vs actual tensors.
//!
//! Spec: docs/specifications/sub/model-layout-provability.md (§36, P0-4)

use std::path::Path;

use super::certify::analyze_config;

/// Run the verify-structure command.
pub fn run(
    contract_dir: &Path,
    config_json: Option<&Path>,
    model_file: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("pv verify-structure — Architecture Structure Verification");
    println!("=========================================================");
    println!();

    // Load architecture schema contract
    let arch_path = find_contract(contract_dir, "apr-architecture-schema-v1");
    let config_path = find_contract(contract_dir, "model-config-algebra-v1");
    let shape_path = find_contract(contract_dir, "tensor-shape-flow-v1");

    let mut found = 0;
    let mut missing = Vec::new();

    for (name, path) in [
        ("apr-architecture-schema-v1", &arch_path),
        ("model-config-algebra-v1", &config_path),
        ("tensor-shape-flow-v1", &shape_path),
    ] {
        if let Some(p) = path {
            let contract = provable_contracts::schema::parse_contract(p)?;
            let eq_count = contract.equations.len();
            let has_assumes = contract
                .equations
                .values()
                .filter(|e| e.assumes.is_some())
                .count();
            let has_guarantees = contract
                .equations
                .values()
                .filter(|e| e.guarantees.is_some())
                .count();
            println!(
                "  ✓ {name}: {eq_count} equations, {has_assumes} assumes, {has_guarantees} guarantees"
            );
            found += 1;
        } else {
            println!("  ✗ {name}: not found");
            missing.push(name);
        }
    }
    println!();

    // Config analysis
    if let Some(cfg) = config_json {
        if cfg.exists() {
            analyze_config(cfg)?;
        } else {
            println!("Config file not found: {}", cfg.display());
        }
    } else {
        println!("No --config provided. Use --config path/to/config.json for structural analysis.");
    }

    // Model file analysis (future)
    if let Some(mf) = model_file {
        println!("Model file: {}", mf.display());
        println!("  ⚠ Model file parsing not yet implemented (P0-4 phase 2)");
        println!("  Planned: enumerate actual tensors, compare shapes to arch-schema");
    }

    println!();
    if missing.is_empty() && found == 3 {
        println!("Result: PASS — all 3 architecture contracts found with composition data");
    } else {
        println!(
            "Result: PARTIAL — {found}/3 contracts found, {} missing",
            missing.len()
        );
    }

    Ok(())
}

fn find_contract(dir: &Path, stem: &str) -> Option<std::path::PathBuf> {
    let direct = dir.join(format!("{stem}.yaml"));
    if direct.exists() {
        return Some(direct);
    }
    // Recursively search subdirectories so contracts nested deeper than one
    // level (e.g. contracts/aprender/foo/bar-v1.yaml) are still located.
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_contract(&path, stem) {
                return Some(found);
            }
        }
    }
    None
}
