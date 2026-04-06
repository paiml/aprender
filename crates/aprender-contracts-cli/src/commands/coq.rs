use std::path::Path;

use provable_contracts::coq_gen::generate_coq_spec;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let output = generate_coq_spec(&contract, stem);
    print!("{output}");

    Ok(())
}
