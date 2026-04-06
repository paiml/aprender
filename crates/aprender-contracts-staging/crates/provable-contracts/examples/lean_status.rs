//! Report Lean 4 proof status across all contracts.
//!
//! Demonstrates the Phase 7 verification hierarchy reporting.
//!
//! Usage:
//!   cargo run --example `lean_status` -- contracts/

use std::path::PathBuf;
use std::process;

use provable_contracts::lean_gen::{format_status_report, lean_status};
use provable_contracts::schema::parse_contract;

fn main() {
    let dir = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: lean_status <contracts-dir/>");
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

    let mut reports = Vec::new();
    for entry in &entries {
        match parse_contract(&entry.path()) {
            Ok(contract) => {
                let report = lean_status(&contract);
                if report.with_lean > 0 {
                    reports.push(report);
                }
            }
            Err(e) => {
                eprintln!("warning: skipping {}: {e}", entry.path().display());
            }
        }
    }

    if reports.is_empty() {
        println!("No Lean proof metadata found in any contracts.");
        println!("Add `lean:` blocks to proof obligations to enable Phase 7 tracking.");
    } else {
        print!("{}", format_status_report(&reports));
    }
}
