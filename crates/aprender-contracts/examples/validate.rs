//! Validate a YAML kernel contract and print any violations.
//!
//! Usage:
//!   cargo run --example validate -- contracts/softmax-kernel-v1.yaml

use std::path::PathBuf;
use std::process;

use provable_contracts::error::Severity;
use provable_contracts::schema::{parse_contract, validate_contract};

fn severity_tag(s: Severity) -> &'static str {
    match s {
        Severity::Error => "ERROR",
        Severity::Warning => "WARN",
        Severity::Info => "INFO",
    }
}

fn main() {
    let path = std::env::args().nth(1).map_or_else(
        || {
            eprintln!("Usage: validate <contract.yaml>");
            process::exit(1);
        },
        PathBuf::from,
    );

    let contract = parse_contract(&path).unwrap_or_else(|e| {
        eprintln!("Failed to parse {}: {e}", path.display());
        process::exit(1);
    });

    println!(
        "Contract: {} v{}",
        contract.metadata.description, contract.metadata.version
    );
    println!("References: {}", contract.metadata.references.len());
    println!();

    let violations = validate_contract(&contract);
    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    let warnings: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Warning)
        .collect();

    if violations.is_empty() {
        println!("No violations found. Contract is valid.");
    } else {
        for v in &violations {
            let loc = v.location.as_deref().unwrap_or("?");
            println!(
                "[{}] {} at {}: {}",
                severity_tag(v.severity),
                v.rule,
                loc,
                v.message
            );
        }
        println!();
        println!("{} error(s), {} warning(s)", errors.len(), warnings.len());
    }

    if !errors.is_empty() {
        process::exit(1);
    }
}
