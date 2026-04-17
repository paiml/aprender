use std::path::Path;

use provable_contracts::infer::{
    ContractSuggestion, InferResult, InferredBinding, format_binding_entry, format_contract_stub,
    infer,
};
use provable_contracts::schema::{Contract, parse_contract};

#[allow(clippy::unnecessary_wraps)]
pub fn run(
    crate_dir: &Path,
    binding_path: &Path,
    contract_dir: &Path,
    top_n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let contracts = load_contracts(contract_dir);
    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let result = infer(crate_dir, binding_path, &refs);

    print_header(crate_dir, &contracts, &result);
    print_matched(&result.matched, top_n);
    print_suggestions(&result.suggestions, top_n);

    if result.matched.is_empty() && result.suggestions.is_empty() {
        println!("No inferences found. All non-trivial functions are bound.");
    }

    Ok(())
}

/// Load all `*.yaml` contracts under `contract_dir` (excluding binding files).
fn load_contracts(contract_dir: &Path) -> Vec<(String, Contract)> {
    let mut contracts = Vec::new();
    let Ok(entries) = std::fs::read_dir(contract_dir) else {
        return contracts;
    };
    for entry in entries.flatten() {
        if let Some(loaded) = load_one_contract(&entry.path()) {
            contracts.push(loaded);
        }
    }
    contracts
}

fn load_one_contract(path: &Path) -> Option<(String, Contract)> {
    if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
        return None;
    }
    if path.to_string_lossy().contains("binding") {
        return None;
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    parse_contract(path).ok().map(|c| (stem, c))
}

fn print_header(crate_dir: &Path, contracts: &[(String, Contract)], result: &InferResult) {
    println!("pv infer — Contract Inference Engine");
    println!("====================================\n");
    println!(
        "Crate: {} | Equations: {} | Unbound: {}",
        crate_dir.display(),
        contracts
            .iter()
            .map(|(_, c)| c.equations.len())
            .sum::<usize>(),
        result.coverage.unbound.len(),
    );
    println!(
        "Reverse coverage: {:.1}% ({}/{})\n",
        result.coverage.coverage_pct, result.coverage.bound_fns, result.coverage.total_pub_fns,
    );
}

fn print_matched(matched: &[InferredBinding], top_n: usize) {
    if matched.is_empty() {
        return;
    }
    println!("=== Inferred Bindings ({}) ===\n", matched.len());
    for (i, m) in matched.iter().take(top_n).enumerate() {
        println!(
            "[{}] {} → {}/{} ({:.0}%, {})",
            i + 1,
            m.function.path,
            m.contract_stem,
            m.equation,
            m.confidence * 100.0,
            m.strategy,
        );
        println!("    {}:{}", m.function.file, m.function.line);
    }
    if matched.len() > top_n {
        println!("    ... and {} more", matched.len() - top_n);
    }

    println!("\n--- Suggested binding.yaml entries ---\n");
    for m in matched.iter().take(top_n) {
        println!("{}\n", format_binding_entry(m));
    }
}

fn print_suggestions(suggestions: &[ContractSuggestion], top_n: usize) {
    if suggestions.is_empty() {
        return;
    }
    println!("=== New Contract Suggestions ({}) ===\n", suggestions.len());
    for (i, s) in suggestions.iter().take(top_n).enumerate() {
        println!(
            "[{}] {} → {}.yaml (Tier {})",
            i + 1,
            s.function.path,
            s.suggested_name,
            s.suggested_tier,
        );
        println!("    {}:{}", s.function.file, s.function.line);
        println!("    {}", s.reason);
    }
    if suggestions.len() > top_n {
        println!("    ... and {} more", suggestions.len() - top_n);
    }

    println!("\n--- Sample contract stub ---\n");
    println!("{}", format_contract_stub(&suggestions[0]));
}
