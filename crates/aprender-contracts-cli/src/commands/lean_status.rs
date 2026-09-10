use std::path::Path;

use crate::contract_walk::collect_corpus;
use provable_contracts::lean_gen::{format_status_report, lean_status};

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // PVL-1 (PMAT-1099): an empty corpus is refused (exit 2). A directory report
    // lists only contracts that carry Lean metadata; a single file is always shown.
    let is_dir = path.is_dir();
    let mut contracts = collect_corpus(path)?;
    contracts.sort_by(|a, b| a.0.cmp(&b.0));
    let reports: Vec<_> = contracts
        .into_iter()
        .map(|(_, c)| lean_status(&c))
        .filter(|r| !is_dir || r.with_lean > 0)
        .collect();

    if reports.is_empty() {
        println!("No Lean proof metadata found in any contracts.");
    } else {
        print!("{}", format_status_report(&reports));
    }

    Ok(())
}
