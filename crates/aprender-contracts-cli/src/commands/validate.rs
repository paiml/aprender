use std::path::Path;

use provable_contracts::error::Severity;
use provable_contracts::schema::{parse_contract, validate_contract};

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let violations = validate_contract(&contract);

    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    let warnings: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Warning)
        .collect();

    for v in &violations {
        println!("{v}");
    }

    println!("\n{} error(s), {} warning(s)", errors.len(), warnings.len());

    if errors.is_empty() {
        println!("Contract is valid.");
        Ok(())
    } else {
        Err(format!("Contract has {} validation error(s)", errors.len()).into())
    }
}
