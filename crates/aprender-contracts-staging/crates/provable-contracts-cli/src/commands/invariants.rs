use std::path::Path;

use provable_contracts::invariant_gen::generate_invariants;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let output = generate_invariants(&contract);

    if output.is_empty() {
        println!("No type_invariants defined in this contract.");
    } else {
        print!("{output}");
    }

    Ok(())
}
