use std::fs;
use std::path::Path;

use provable_contracts::lean_gen::generate_lean_files;
use provable_contracts::schema::parse_contract;

pub fn run(path: &Path, output_dir: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let contract = parse_contract(path)?;
    let files = generate_lean_files(&contract);

    if files.is_empty() {
        println!(
            "No Lean metadata found in {}. Add `lean:` blocks to proof obligations.",
            path.display()
        );
        return Ok(());
    }

    if let Some(dir) = output_dir {
        for f in &files {
            let full = dir.join(&f.path);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&full, &f.content)?;
            println!("  {}", full.display());
        }
        println!("\nGenerated {} Lean files.", files.len());
    } else {
        // No output dir → print to stdout
        for f in &files {
            println!("// === {} ===\n", f.path);
            print!("{}", f.content);
            println!();
        }
    }

    Ok(())
}
