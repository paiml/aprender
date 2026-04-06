use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::explain::{
    explain_contract, explain_contract_json, explain_contract_markdown,
};
use provable_contracts::schema::parse_contract;

pub fn run(
    path: &Path,
    binding_path: Option<&Path>,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    let output = match format {
        "markdown" | "md" => explain_contract_markdown(&contract, stem, binding.as_ref()),
        "json" => explain_contract_json(&contract, stem, binding.as_ref()),
        _ => explain_contract(&contract, stem, binding.as_ref()),
    };
    print!("{output}");

    Ok(())
}
