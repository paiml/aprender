use std::path::Path;

use provable_contracts::binding::{BindingRegistry, parse_binding};
use provable_contracts::generate::{GeneratedFiles, generate_all};
use provable_contracts::readme_gen::{generate_ci_workflow, generate_readme};
use provable_contracts::schema::{Contract, parse_contract};

pub fn run(
    contract: &Path,
    output_dir: &Path,
    binding_path: Option<&Path>,
    readme: bool,
    ci: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let c = parse_contract(contract)?;
    let binding = load_binding(binding_path)?;
    let stem = contract
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("contract");

    let result = generate_all(&c, stem, output_dir, binding.as_ref())?;
    print_generated_files(&result, output_dir);

    if readme {
        write_readme(&c, stem, output_dir, binding.as_ref())?;
    }
    if ci {
        write_ci_workflow(output_dir, binding.as_ref())?;
    }
    Ok(())
}

fn load_binding(
    binding_path: Option<&Path>,
) -> Result<Option<BindingRegistry>, Box<dyn std::error::Error>> {
    match binding_path {
        Some(bp) => Ok(Some(parse_binding(bp)?)),
        None => Ok(None),
    }
}

fn print_generated_files(result: &GeneratedFiles, output_dir: &Path) {
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
}

fn write_readme(
    c: &Contract,
    stem: &str,
    output_dir: &Path,
    binding: Option<&BindingRegistry>,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(reg) = binding else {
        eprintln!("warning: --readme requires --binding to generate coverage report");
        return Ok(());
    };
    let contracts = vec![(stem.to_string(), c)];
    let readme_content = generate_readme(&contracts, reg);
    let readme_path = output_dir.join("CONTRACT-README.md");
    std::fs::write(&readme_path, &readme_content)?;
    println!(
        "  CONTRACT-README.md (readme, {} bytes)",
        readme_content.len()
    );
    Ok(())
}

fn write_ci_workflow(
    output_dir: &Path,
    binding: Option<&BindingRegistry>,
) -> Result<(), Box<dyn std::error::Error>> {
    let project_name = binding.map_or("project", |b| b.target_crate.as_str());
    let ci_content = generate_ci_workflow(project_name);
    let ci_dir = output_dir.join(".github").join("workflows");
    std::fs::create_dir_all(&ci_dir)?;
    let ci_path = ci_dir.join("contracts.yml");
    std::fs::write(&ci_path, &ci_content)?;
    println!(
        "  .github/workflows/contracts.yml (ci, {} bytes)",
        ci_content.len()
    );
    Ok(())
}
