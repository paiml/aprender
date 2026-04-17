use std::path::Path;

use provable_contracts::book_gen::{generate_contract_page, update_summary};
use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{Contract, parse_contract};

pub fn run(
    contract_dir: &Path,
    output_dir: &Path,
    update_summary_flag: bool,
    summary_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let contracts = load_contracts(contract_dir)?;

    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    std::fs::create_dir_all(output_dir)?;
    let generated = write_contract_pages(&contracts, output_dir, &graph)?;

    if update_summary_flag {
        write_summary(&contracts, summary_path)?;
    }

    print_manifest(&generated, output_dir);
    Ok(())
}

fn load_contracts(
    contract_dir: &Path,
) -> Result<Vec<(String, Contract)>, Box<dyn std::error::Error>> {
    let mut contracts = Vec::new();
    for entry in std::fs::read_dir(contract_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        match parse_contract(&path) {
            Ok(c) => contracts.push((stem, c)),
            Err(e) => eprintln!("warning: skipping {}: {e}", path.display()),
        }
    }
    contracts.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(contracts)
}

fn write_contract_pages(
    contracts: &[(String, Contract)],
    output_dir: &Path,
    graph: &provable_contracts::graph::DependencyGraph,
) -> Result<Vec<(String, usize)>, Box<dyn std::error::Error>> {
    let mut generated = Vec::new();
    for (stem, contract) in contracts {
        let page = generate_contract_page(contract, stem, graph);
        let out_path = output_dir.join(format!("{stem}.md"));
        std::fs::write(&out_path, &page)?;
        generated.push((stem.clone(), page.len()));
    }
    Ok(generated)
}

fn write_summary(
    contracts: &[(String, Contract)],
    summary_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = summary_path.unwrap_or_else(|| Path::new("book/src/SUMMARY.md"));
    let existing = std::fs::read_to_string(summary)?;
    let stems: Vec<&str> = contracts.iter().map(|(s, _)| s.as_str()).collect();
    let updated = update_summary(&existing, &stems);
    std::fs::write(summary, &updated)?;
    println!("Updated {}", summary.display());
    Ok(())
}

fn print_manifest(generated: &[(String, usize)], output_dir: &Path) {
    println!("Generated {} contract pages:", generated.len());
    for (stem, bytes) in generated {
        println!(
            "  {output_dir}/{stem}.md ({bytes} bytes)",
            output_dir = output_dir.display()
        );
    }
}
