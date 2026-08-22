//! Example: Lean 4 proof status across foundation kernels.
//!
//! Demonstrates the Lean integration:
//! - Lean definition + theorem generation from YAML contracts
//! - L4 proof coverage tracking (proved vs sorry)
//! - Lean-aware scoring dimension
//!
//! Run from the workspace root:
//!   cargo run --example `lean_proofs`

use std::path::Path;

use provable_contracts::lean_gen;
use provable_contracts::schema::parse_contract;
use provable_contracts::scoring;

fn main() {
    let contracts_dir = Path::new("contracts");

    // Contracts with Lean proofs
    let lean_contracts = [
        "softmax-kernel-v1",
        "rmsnorm-kernel-v1",
        "silu-kernel-v1",
        "cross-entropy-kernel-v1",
        "layernorm-kernel-v1",
        "transpose-kernel-v1",
    ];

    println!("=== Lean 4 Proof Status ===\n");

    let mut reports = Vec::new();
    for stem in &lean_contracts {
        let path = contracts_dir.join(format!("{stem}.yaml"));
        let contract = parse_contract(&path).unwrap();
        let report = lean_gen::lean_status(&contract);
        reports.push(report);
    }
    print!("{}", lean_gen::format_status_report(&reports));

    println!("\n=== Score Impact (Lean Dimension) ===\n");
    for stem in &lean_contracts {
        let path = contracts_dir.join(format!("{stem}.yaml"));
        let contract = parse_contract(&path).unwrap();
        let score = scoring::score_contract(&contract, None, stem);
        println!("{stem}");
        println!(
            "  Score: {:.2} (Grade {}) — Lean: {:.2}",
            score.composite, score.grade, score.lean_coverage
        );
    }

    println!("\n=== Generated Lean for softmax ===\n");
    let softmax = parse_contract(&contracts_dir.join("softmax-kernel-v1.yaml")).unwrap();
    let files = lean_gen::generate_lean_files(&softmax);
    for f in &files {
        println!("--- {} ---", f.path);
        // Show first 10 lines of each file
        for line in f.content.lines().take(10) {
            println!("{line}");
        }
        println!("  ...\n");
    }
    println!("{} Lean files generated for softmax", files.len());
}
