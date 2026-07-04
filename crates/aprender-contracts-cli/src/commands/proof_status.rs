use std::path::Path;

use provable_contracts::binding::parse_binding;
use provable_contracts::obligation_matrix::{format_obligation_table, obligation_matrix};
use provable_contracts::proof_status::{format_text, proof_status_report};
use provable_contracts::schema::{parse_contract, ContractKind};

use crate::contract_walk::collect_contracts;

pub fn run(
    path: &Path,
    binding_path: Option<&Path>,
    verify_root: Option<&Path>,
    format: &str,
    table: bool,
    kind_filter: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let binding = match binding_path {
        // L5 gate: when --verify-bindings is set, downgrade any `implemented`
        // binding whose function is absent from source, so L5 requires bindings
        // that are VERIFIED as implemented rather than merely self-declared.
        Some(bp) => Some(match verify_root {
            Some(root) => parse_binding(bp)?.verified(root),
            None => parse_binding(bp)?,
        }),
        None => None,
    };

    let kind = kind_filter.map(parse_kind).transpose()?;

    // Collect contracts (single file or directory tree)
    let mut contracts = Vec::new();
    if path.is_dir() {
        collect_contracts(path, &mut contracts);
    } else {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let c = parse_contract(path)?;
        contracts.push((stem, c));
    }

    if let Some(k) = kind {
        contracts.retain(|(_, c)| c.kind() == k);
    }

    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let include_classes = contracts.len() > 1;
    let report = proof_status_report(&refs, binding.as_ref(), include_classes);

    if format == "json" {
        let json = serde_json::to_string_pretty(&report)?;
        println!("{json}");
    } else {
        print!("{}", format_text(&report));
        // Append kind breakdown when showing >1 contract.
        if contracts.len() > 1 {
            print_kind_breakdown(&contracts);
        }
    }

    if table {
        let matrices = obligation_matrix(&refs);
        print!("{}", format_obligation_table(&matrices));
    }

    Ok(())
}

fn print_kind_breakdown(contracts: &[(String, provable_contracts::schema::Contract)]) {
    let mut counts = std::collections::BTreeMap::<ContractKind, usize>::new();
    for (_, c) in contracts {
        *counts.entry(c.kind()).or_insert(0) += 1;
    }
    // Only print if there's > 1 kind represented.
    if counts.len() < 2 {
        return;
    }
    println!();
    print!("By kind:");
    for (kind, count) in &counts {
        print!("  {kind}={count}");
    }
    println!();
}

fn parse_kind(s: &str) -> Result<ContractKind, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "kernel" => Ok(ContractKind::Kernel),
        "registry" => Ok(ContractKind::Registry),
        "model-family" | "modelfamily" => Ok(ContractKind::ModelFamily),
        "pattern" => Ok(ContractKind::Pattern),
        "schema" => Ok(ContractKind::Schema),
        other => Err(format!(
            "invalid --kind value '{other}': expected one of \
             kernel, registry, model-family, pattern, schema"
        )
        .into()),
    }
}
