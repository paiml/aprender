use std::path::Path;

use provable_contracts::lean_gen::{format_status_report, lean_status};
use provable_contracts::schema::parse_contract;

use crate::contract_walk::collect_contracts;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let reports = if path.is_dir() {
        let mut contracts = Vec::new();
        collect_contracts(path, &mut contracts);
        contracts.sort_by(|a, b| a.0.cmp(&b.0));
        contracts
            .into_iter()
            .map(|(_, c)| lean_status(&c))
            .filter(|r| r.with_lean > 0)
            .collect()
    } else {
        let contract = parse_contract(path)?;
        vec![lean_status(&contract)]
    };

    if reports.is_empty() {
        println!("No Lean proof metadata found in any contracts.");
    } else {
        print!("{}", format_status_report(&reports));
    }

    Ok(())
}
