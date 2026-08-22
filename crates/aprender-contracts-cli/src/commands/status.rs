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
    // #2504: `contracts/publish-workspace-v1.yaml` holds four FALSIFY-PUB-*
    // entries under a top-level `falsification:` key that is NOT
    // `falsification_tests`. Before the schema captured that block, this
    // command printed "Falsification tests: 0" and stopped — the reader was
    // told the count and never told where the entries went. Say it out loud.
    let legacy = contract.legacy_falsification_entries();
    if legacy > 0 {
        println!(
            "  ...plus {legacy} entr{} in the legacy top-level `falsification:` block, \
             which NO pv gate enforces{}",
            if legacy == 1 { "y" } else { "ies" },
            if contract.falsification_tests.is_empty() {
                " — this contract is INERT: it reads as enforced and enforces nothing"
            } else {
                ""
            }
        );
    }
    println!("Kani harnesses: {}", contract.kani_harnesses.len());

    if let Some(ref gate) = contract.qa_gate {
        println!("QA gate: {} ({})", gate.name, gate.id);
    } else {
        println!("QA gate: not defined");
    }

    Ok(())
}
