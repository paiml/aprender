use std::path::Path;

use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;

    println!(
        "Contract: {} v{}",
        contract.metadata.description, contract.metadata.version
    );
    println!("References: {}", contract.metadata.references.len());
    println!("Equations: {}", contract.equations.len());
    println!("Proof obligations: {}", contract.proof_obligations.len());
    println!(
        "Falsification tests: {}",
        contract.falsification_tests.len()
    );
    println!("Kani harnesses: {}", contract.kani_harnesses.len());

    if let Some(ref gate) = contract.qa_gate {
        println!("QA gate: {} ({})", gate.name, gate.id);
    } else {
        println!("QA gate: not defined");
    }

    Ok(())
}
