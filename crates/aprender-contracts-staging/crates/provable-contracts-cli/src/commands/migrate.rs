use std::path::Path;

pub fn run(contract_dir: &Path, dry_run: bool) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("pv migrate — contract schema migration");
    eprintln!("=========================================\n");
    let mut count = 0;
    for entry in std::fs::read_dir(contract_dir)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let content = std::fs::read_to_string(&path)?;
            let needs = !content.contains("metadata:")
                || (!content.contains("proof_obligations:") && !content.contains("registry: true"));
            if needs {
                if dry_run {
                    eprintln!("  [MIGRATE] {}", path.display());
                } else {
                    eprintln!(
                        "  [MIGRATE] {} (auto-fix not yet implemented)",
                        path.display()
                    );
                }
                count += 1;
            }
        }
    }
    if count == 0 {
        eprintln!(
            "  All contracts in {} use current schema.",
            contract_dir.display()
        );
    } else {
        eprintln!("\n{count} contract(s) need migration.");
        if dry_run {
            eprintln!("Run without --dry-run to apply fixes.");
        }
    }
    Ok(())
}
