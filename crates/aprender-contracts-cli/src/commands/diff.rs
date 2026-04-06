use std::path::Path;

use provable_contracts::diff::{diff_contracts, is_identical};
use provable_contracts::schema::parse_contract;

pub fn run(old_path: &Path, new_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let old = parse_contract(old_path)?;
    let new = parse_contract(new_path)?;

    let diff = diff_contracts(&old, &new);

    if is_identical(&diff) {
        println!("Contracts are identical.");
        return Ok(());
    }

    println!(
        "Contract diff: v{} → v{}",
        diff.old_version, diff.new_version
    );
    println!("Suggested bump: {}", diff.suggested_bump);
    println!();

    for section in &diff.sections {
        if section.is_empty() {
            continue;
        }
        println!("  {}:", section.section);
        for a in &section.added {
            println!("    + {a}");
        }
        for r in &section.removed {
            println!("    - {r}");
        }
        for c in &section.changed {
            println!("    ~ {c}");
        }
    }

    Ok(())
}
