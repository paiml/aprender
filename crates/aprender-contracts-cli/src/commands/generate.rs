use std::path::Path;

use provable_contracts::binding::{parse_binding, BindingRegistry};
use provable_contracts::generate::{generate_all, plan_all, GeneratedFiles};
use provable_contracts::readme_gen::{generate_ci_workflow, generate_readme};
use provable_contracts::schema::{parse_contract, Contract};

pub fn run(
    contract: &Path,
    output_dir: &Path,
    binding_path: Option<&Path>,
    readme: bool,
    ci: bool,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let c = parse_contract(contract)?;
    let binding = load_binding(binding_path)?;
    let stem = contract
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("contract");

    if dry_run {
        return preview(&c, stem, output_dir, binding.as_ref(), readme, ci);
    }

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

fn verb(path: &Path) -> &'static str {
    if path.exists() {
        "overwrite"
    } else {
        "create"
    }
}

/// `--dry-run`: render the artifacts in memory (`plan_all`) and report the paths a real run would
/// write under `output_dir`. No file or directory is created anywhere, `TMPDIR` included.
fn preview(
    c: &Contract,
    stem: &str,
    output_dir: &Path,
    binding: Option<&BindingRegistry>,
    readme: bool,
    ci: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = plan_all(c, stem, output_dir, binding);

    let mut planned: Vec<(std::path::PathBuf, String, usize)> = result
        .files
        .iter()
        .map(|f| (f.relative_path.clone(), f.kind.to_string(), f.bytes))
        .collect();
    if readme {
        if let Some(reg) = binding {
            let body = generate_readme(&[(stem.to_string(), c)], reg);
            planned.push(("CONTRACT-README.md".into(), "readme".into(), body.len()));
        } else {
            eprintln!("warning: --readme requires --binding to generate coverage report");
        }
    }
    if ci {
        let name = binding.map_or("project", |b| b.target_crate.as_str());
        let body = generate_ci_workflow(name);
        planned.push((
            ".github/workflows/contracts.yml".into(),
            "ci".into(),
            body.len(),
        ));
    }

    println!(
        "Dry run: {} files would be written in {} (nothing written):",
        planned.len(),
        output_dir.display()
    );
    for (rel, kind, bytes) in &planned {
        let full = output_dir.join(rel);
        println!(
            "  would {} {} ({kind}, {bytes} bytes)",
            verb(&full),
            rel.display()
        );
    }
    Ok(())
}
