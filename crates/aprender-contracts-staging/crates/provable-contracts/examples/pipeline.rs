//! Run the full provable-contracts pipeline on all contracts in a directory.
//!
//! Demonstrates: parse -> validate -> audit -> report for each contract.
//!
//! Usage:
//!   cargo run --example pipeline -- contracts/

use std::path::PathBuf;
use std::process;

use provable_contracts::audit::audit_contract;
use provable_contracts::error::Severity;
use provable_contracts::schema::{parse_contract, validate_contract};

fn main() {
    let dir = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: pipeline <contracts-dir/>");
            process::exit(1);
        },
        PathBuf::from,
    );

    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read {}: {e}", dir.display());
            process::exit(1);
        })
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "yaml" || ext == "yml")
        })
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

    if entries.is_empty() {
        eprintln!("No YAML files found in {}", dir.display());
        process::exit(1);
    }

    let header = format!(
        "{:<35} {:>4} {:>4} {:>5} {:>5} {:>5}  {}",
        "CONTRACT", "EQS", "OBLS", "FALS", "KANI", "AUDIT", "STATUS"
    );
    println!("{header}");
    println!("{}", "-".repeat(85));

    let mut total_errors = 0;

    for entry in &entries {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy();

        let contract = match parse_contract(&path) {
            Ok(c) => c,
            Err(e) => {
                println!("{filename:<35} PARSE ERROR: {e}");
                total_errors += 1;
                continue;
            }
        };

        let violations = validate_contract(&contract);
        let error_count = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .count();

        let report = audit_contract(&contract);
        let audit_findings = report.violations.len();

        let status = if error_count > 0 {
            total_errors += error_count;
            format!("{error_count} errors")
        } else if audit_findings > 0 {
            format!("{audit_findings} findings")
        } else {
            "OK".to_string()
        };

        println!(
            "{filename:<35} {:>4} {:>4} {:>5} {:>5} {:>5}  {status}",
            contract.equations.len(),
            contract.proof_obligations.len(),
            contract.falsification_tests.len(),
            contract.kani_harnesses.len(),
            audit_findings,
        );
    }

    println!("{}", "-".repeat(85));
    if total_errors > 0 {
        println!("{total_errors} total error(s)");
        process::exit(1);
    }
    println!("All contracts valid.");
}
