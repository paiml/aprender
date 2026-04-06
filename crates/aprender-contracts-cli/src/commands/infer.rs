use std::path::Path;

use provable_contracts::infer::{format_binding_entry, format_contract_stub, infer};
use provable_contracts::schema::parse_contract;

#[allow(clippy::unnecessary_wraps)]
pub fn run(
    crate_dir: &Path,
    binding_path: &Path,
    contract_dir: &Path,
    top_n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load all contracts
    let mut contracts = Vec::new();
    if let Ok(entries) = std::fs::read_dir(contract_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml")
                && !path.to_string_lossy().contains("binding")
            {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                if let Ok(c) = parse_contract(&path) {
                    contracts.push((stem, c));
                }
            }
        }
    }

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let result = infer(crate_dir, binding_path, &refs);

    // Header
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

    // Matched bindings
    if !result.matched.is_empty() {
        println!("=== Inferred Bindings ({}) ===\n", result.matched.len());
        for (i, m) in result.matched.iter().take(top_n).enumerate() {
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
        if result.matched.len() > top_n {
            println!("    ... and {} more", result.matched.len() - top_n);
        }

        println!("\n--- Suggested binding.yaml entries ---\n");
        for m in result.matched.iter().take(top_n) {
            println!("{}\n", format_binding_entry(m));
        }
    }

    // New contract suggestions
    if !result.suggestions.is_empty() {
        println!(
            "=== New Contract Suggestions ({}) ===\n",
            result.suggestions.len()
        );
        for (i, s) in result.suggestions.iter().take(top_n).enumerate() {
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
        if result.suggestions.len() > top_n {
            println!("    ... and {} more", result.suggestions.len() - top_n);
        }

        if !result.suggestions.is_empty() {
            println!("\n--- Sample contract stub ---\n");
            println!("{}", format_contract_stub(&result.suggestions[0]));
        }
    }

    if result.matched.is_empty() && result.suggestions.is_empty() {
        println!("No inferences found. All non-trivial functions are bound.");
    }

    Ok(())
}
