//! Run a traceability audit on a contract, optionally with bindings.
//!
//! Usage:
//!   cargo run --example audit -- contracts/softmax-kernel-v1.yaml
//!   cargo run --example audit -- contracts/softmax-kernel-v1.yaml contracts/aprender/binding.yaml

use std::path::PathBuf;
use std::process;

use provable_contracts::audit::{audit_binding, audit_contract};
use provable_contracts::binding::parse_binding;
use provable_contracts::error::Severity;
use provable_contracts::schema::parse_contract;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let contract_path = args.get(1).map_or_else(
        || {
            eprintln!("Usage: audit <contract.yaml> [binding.yaml]");
            process::exit(1);
        },
        PathBuf::from,
    );

    let contract = parse_contract(&contract_path).unwrap_or_else(|e| {
        eprintln!("Failed to parse {}: {e}", contract_path.display());
        process::exit(1);
    });

    // Traceability audit
    let report = audit_contract(&contract);
    println!("Traceability Audit");
    println!("==================");
    println!("Equations:           {}", report.equations);
    println!("Proof obligations:   {}", report.obligations);
    println!("Falsification tests: {}", report.falsification_tests);
    println!("Kani harnesses:      {}", report.kani_harnesses);
    println!();

    let has_errors = report
        .violations
        .iter()
        .any(|v| v.severity == Severity::Error);
    if report.violations.is_empty() {
        println!("No audit findings.");
    } else {
        for v in &report.violations {
            println!("  [{:?}] {}: {}", v.severity, v.rule, v.message);
        }
    }

    // Binding audit (optional)
    if let Some(binding_path) = args.get(2) {
        println!();
        let registry = parse_binding(PathBuf::from(binding_path).as_path()).unwrap_or_else(|e| {
            eprintln!("Failed to parse binding: {e}");
            process::exit(1);
        });
        let contract_file = contract_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let contracts = vec![(contract_file.as_str(), &contract)];
        let bind_report = audit_binding(&contracts, &registry);
        println!("Binding Audit (target: {})", registry.target_crate);
        println!("=============");
        println!("Bound equations:     {}", bind_report.bound_equations);
        println!("Implemented:         {}", bind_report.implemented);
        println!("Partial:             {}", bind_report.partial);
        println!("Not implemented:     {}", bind_report.not_implemented);
        println!("Obligations covered: {}", bind_report.covered_obligations);
        println!();
        if bind_report.violations.is_empty() {
            println!("No binding gaps found.");
        } else {
            for v in &bind_report.violations {
                println!("  [{:?}] {}: {}", v.severity, v.rule, v.message);
            }
        }
    }

    if has_errors {
        process::exit(1);
    }
}
