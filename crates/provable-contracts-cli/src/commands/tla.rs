use std::path::Path;

use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{Contract, parse_contract};
use provable_contracts::tla_gen::generate_tla_module;

pub fn run(contract_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut contracts: Vec<(String, Contract)> = Vec::new();

    let entries = std::fs::read_dir(contract_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            match parse_contract(&path) {
                Ok(c) => contracts.push((stem, c)),
                Err(e) => {
                    eprintln!("warning: skipping {}: {e}", path.display());
                }
            }
        }
    }

    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    let module_name = contract_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("Contracts");
    let module_name = module_name
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>();
    let module_name = if module_name.is_empty() {
        "Contracts".to_string()
    } else {
        // Capitalize first letter
        let mut chars = module_name.chars();
        match chars.next() {
            None => "Contracts".to_string(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    };

    let output = generate_tla_module(&module_name, &refs, &graph);
    print!("{output}");

    Ok(())
}
