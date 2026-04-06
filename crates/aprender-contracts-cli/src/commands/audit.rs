use std::path::Path;

use provable_contracts::audit::{audit_binding, audit_contract};
use provable_contracts::binding::parse_binding;
use provable_contracts::error::Severity;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path, binding_path: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;

    // Standard traceability audit
    let report = audit_contract(&contract);

    println!("Traceability Audit");
    println!("==================");
    println!("Equations:          {}", report.equations);
    println!("Proof obligations:  {}", report.obligations);
    println!("Falsification tests: {}", report.falsification_tests);
    println!("Kani harnesses:     {}", report.kani_harnesses);
    println!("Type invariants:    {}", contract.type_invariants.len());

    // Lean status
    let lean_proved = contract
        .verification_summary
        .as_ref()
        .map_or(0, |vs| vs.l4_lean_proved);
    if lean_proved > 0 {
        println!(
            "Lean proved:        {}/{}",
            lean_proved,
            contract
                .verification_summary
                .as_ref()
                .map_or(0, |vs| vs.total_obligations)
        );
    }

    // Coq status
    if let Some(ref spec) = contract.coq_spec {
        let total = spec.obligations.len();
        let proved = spec
            .obligations
            .iter()
            .filter(|o| o.status == "proved")
            .count();
        let admitted = spec
            .obligations
            .iter()
            .filter(|o| o.status == "admitted")
            .count();
        let stubs = total - proved - admitted;
        println!(
            "Coq ({}):{} {proved} proved, {admitted} admitted, {stubs} stub",
            spec.module,
            if total > 0 {
                format!("  {total} obligations —")
            } else {
                " no obligation links".to_string()
            }
        );
    }

    println!();

    if report.violations.is_empty() {
        println!("No audit findings.");
    } else {
        for v in &report.violations {
            println!("{v}");
        }
    }

    let errors = report
        .violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count();

    // Binding audit (if --binding provided)
    if let Some(bp) = binding_path {
        let binding = parse_binding(bp)?;

        let contract_file = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let binding_report = audit_binding(&[(contract_file, &contract)], &binding);

        println!();
        println!("Binding Audit");
        println!("=============");
        println!("Total equations:    {}", binding_report.total_equations);
        println!("Bound equations:    {}", binding_report.bound_equations);
        println!("Implemented:        {}", binding_report.implemented);
        println!("Partial:            {}", binding_report.partial);
        println!("Not implemented:    {}", binding_report.not_implemented);
        println!("Obligations total:  {}", binding_report.total_obligations);
        println!(
            "Obligations covered: {}",
            binding_report.covered_obligations
        );
        println!();

        if binding_report.violations.is_empty() {
            println!("No binding gaps found.");
        } else {
            for v in &binding_report.violations {
                println!("{v}");
            }
        }

        let binding_errors = binding_report
            .violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .count();

        if errors + binding_errors > 0 {
            return Err(format!("Audit found {} error(s)", errors + binding_errors).into());
        }
    } else if errors > 0 {
        return Err(format!("Audit found {errors} error(s)").into());
    }

    Ok(())
}
