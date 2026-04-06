use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::generate::generate_all;
use provable_contracts::readme_gen::{generate_ci_workflow, generate_readme};
use provable_contracts::schema::parse_contract;

pub fn run(
    contract: &Path,
    output_dir: &Path,
    binding_path: Option<&Path>,
    readme: bool,
    ci: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let c = parse_contract(contract)?;

    let binding = match binding_path {
        Some(bp) => Some(parse_binding(bp)?),
        None => None,
    };

    let stem = contract
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("contract");

    let result = generate_all(&c, stem, output_dir, binding.as_ref())?;

    println!(
        "Generated {} files in {}:",
        result.files.len(),
        output_dir.display()
    );
    for f in &result.files {
        println!(
            "  {} ({}, {} bytes)",
            f.relative_path.display(),
            f.kind,
            f.bytes
        );
    }

    // Generate CONTRACT-README.md
    if readme {
        if let Some(ref reg) = binding {
            let contracts = vec![(stem.to_string(), &c)];
            let readme_content = generate_readme(&contracts, reg);
            let readme_path = output_dir.join("CONTRACT-README.md");
            std::fs::write(&readme_path, &readme_content)?;
            println!(
                "  CONTRACT-README.md (readme, {} bytes)",
                readme_content.len()
            );
        } else {
            eprintln!("warning: --readme requires --binding to generate coverage report");
        }
    }

    // Generate CI workflow
    if ci {
        let project_name = binding
            .as_ref()
            .map_or("project", |b| b.target_crate.as_str());
        let ci_content = generate_ci_workflow(project_name);
        let ci_dir = output_dir.join(".github").join("workflows");
        std::fs::create_dir_all(&ci_dir)?;
        let ci_path = ci_dir.join("contracts.yml");
        std::fs::write(&ci_path, &ci_content)?;
        println!(
            "  .github/workflows/contracts.yml (ci, {} bytes)",
            ci_content.len()
        );
    }

    Ok(())
}
