use std::path::{Path, PathBuf};

pub fn run(contract_dir: &Path, dry_run: bool) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("pv migrate — contract schema migration");
    eprintln!("=========================================\n");

    let count = scan_and_report(contract_dir, dry_run)?;
    print_summary(contract_dir, dry_run, count);
    Ok(())
}

fn scan_and_report(contract_dir: &Path, dry_run: bool) -> std::io::Result<usize> {
    let mut count = 0;
    for entry in std::fs::read_dir(contract_dir)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        if needs_migration(&path)? {
            report_migration(&path, dry_run);
            count += 1;
        }
    }
    Ok(count)
}

fn needs_migration(path: &PathBuf) -> std::io::Result<bool> {
    let content = std::fs::read_to_string(path)?;
    Ok(!content.contains("metadata:")
        || (!content.contains("proof_obligations:") && !content.contains("registry: true")))
}

fn report_migration(path: &Path, dry_run: bool) {
    if dry_run {
        eprintln!("  [MIGRATE] {}", path.display());
    } else {
        eprintln!(
            "  [MIGRATE] {} (auto-fix not yet implemented)",
            path.display()
        );
    }
}

fn print_summary(contract_dir: &Path, dry_run: bool, count: usize) {
    if count == 0 {
        eprintln!(
            "  All contracts in {} use current schema.",
            contract_dir.display()
        );
        return;
    }
    eprintln!("\n{count} contract(s) need migration.");
    if dry_run {
        eprintln!("Run without --dry-run to apply fixes.");
    }
}
