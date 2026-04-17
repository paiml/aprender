use std::path::Path;

use provable_contracts::audit::{audit_binding, audit_contract};
use provable_contracts::binding::parse_binding;
use provable_contracts::error::Severity;
use provable_contracts::schema::{Contract, parse_contract};

pub fn run(path: &Path, binding_path: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;

    let report = audit_contract(&contract);
    print_traceability_header(&contract, &report);
    print_lean_status(&contract);
    print_coq_status(&contract);
    print_violations(&report.violations);

    let errors = count_errors(&report.violations);
    let binding_errors = run_binding_audit(path, &contract, binding_path)?;

    let total = errors + binding_errors;
    if total > 0 {
        return Err(format!("Audit found {total} error(s)").into());
    }
    Ok(())
}

fn print_traceability_header(contract: &Contract, report: &provable_contracts::audit::AuditReport) {
    println!("Traceability Audit");
    println!("==================");
    println!("Equations:          {}", report.equations);
    println!("Proof obligations:  {}", report.obligations);
    println!("Falsification tests: {}", report.falsification_tests);
    println!("Kani harnesses:     {}", report.kani_harnesses);
    println!("Type invariants:    {}", contract.type_invariants.len());
}

fn print_lean_status(contract: &Contract) {
    let lean_proved = contract
        .verification_summary
        .as_ref()
        .map_or(0, |vs| vs.l4_lean_proved);
    if lean_proved == 0 {
        return;
    }
    let total = contract
        .verification_summary
        .as_ref()
        .map_or(0, |vs| vs.total_obligations);
    println!("Lean proved:        {lean_proved}/{total}");
}

fn print_coq_status(contract: &Contract) {
    let Some(spec) = contract.coq_spec.as_ref() else {
        return;
    };
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
    let suffix = if total > 0 {
        format!("  {total} obligations —")
    } else {
        " no obligation links".to_string()
    };
    println!(
        "Coq ({}):{suffix} {proved} proved, {admitted} admitted, {stubs} stub",
        spec.module
    );
}

fn print_violations(violations: &[provable_contracts::error::Violation]) {
    println!();
    if violations.is_empty() {
        println!("No audit findings.");
    } else {
        for v in violations {
            println!("{v}");
        }
    }
}

fn count_errors(violations: &[provable_contracts::error::Violation]) -> usize {
    violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count()
}

fn run_binding_audit(
    path: &Path,
    contract: &Contract,
    binding_path: Option<&Path>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let Some(bp) = binding_path else {
        return Ok(0);
    };
    let binding = parse_binding(bp)?;
    let contract_file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    let binding_report = audit_binding(&[(contract_file, contract)], &binding);

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
    Ok(count_errors(&binding_report.violations))
}
