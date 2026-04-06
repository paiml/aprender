use std::path::Path;

use provable_contracts::book_gen::{generate_contract_page, update_summary};
use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::parse_contract;

pub fn run(
    contract_dir: &Path,
    output_dir: &Path,
    update_summary_flag: bool,
    summary_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load all contracts
    let mut contracts = Vec::new();
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

    // Build dependency graph
    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Generate pages
    let mut generated = Vec::new();
    for (stem, contract) in &contracts {
        let page = generate_contract_page(contract, stem, &graph);
        let out_path = output_dir.join(format!("{stem}.md"));
        std::fs::write(&out_path, &page)?;
        generated.push((stem.clone(), page.len()));
    }

    // Update SUMMARY.md if requested
    if update_summary_flag {
        let summary = summary_path.unwrap_or_else(|| Path::new("book/src/SUMMARY.md"));
        let existing = std::fs::read_to_string(summary)?;
        let stems: Vec<&str> = contracts.iter().map(|(s, _)| s.as_str()).collect();
        let updated = update_summary(&existing, &stems);
        std::fs::write(summary, &updated)?;
        println!("Updated {}", summary.display());
    }

    // Print manifest
    println!("Generated {} contract pages:", generated.len());
    for (stem, bytes) in &generated {
        println!(
            "  {output_dir}/{stem}.md ({bytes} bytes)",
            output_dir = output_dir.display()
        );
    }

    Ok(())
}
