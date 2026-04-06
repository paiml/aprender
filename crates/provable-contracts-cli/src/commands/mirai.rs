use std::path::Path;

use provable_contracts::mirai_gen::generate_mirai_annotations;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let output = generate_mirai_annotations(&contract, stem);
    print!("{output}");

    Ok(())
}
