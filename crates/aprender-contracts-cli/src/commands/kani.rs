use std::path::Path;

use provable_contracts::kani_gen::generate_kani_harnesses;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    print!("{}", generate_kani_harnesses(&contract));
    Ok(())
}
