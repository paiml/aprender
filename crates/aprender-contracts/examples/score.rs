//! Example: Score contracts using the scoring module.
//!
//! ```bash
//! cargo run --example score
//! ```

use std::path::Path;

fn main() {
    let contracts_dir = Path::new("contracts");

    let mut entries: Vec<_> = std::fs::read_dir(contracts_dir)
        .expect("contracts/ directory must exist")
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("yaml"))
        .collect();
    entries.sort_by_key(std::fs::DirEntry::path);

    let mut scores = Vec::new();
    for entry in &entries {
        let path = entry.path();
        let Ok(contract) = provable_contracts::schema::parse_contract(&path) else {
            continue;
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let score = provable_contracts::scoring::score_contract(&contract, None, stem);
        scores.push(score);
    }

    // Sort by composite score descending
    scores.sort_by(|a, b| b.composite.partial_cmp(&a.composite).unwrap());

    println!("=== Top 10 Contracts by Score ===\n");
    for s in scores.iter().take(10) {
        print!("{s}");
    }

    println!("\n=== Bottom 5 Contracts (improvement targets) ===\n");
    for s in scores.iter().rev().take(5) {
        print!("{s}");
    }

    #[allow(clippy::cast_precision_loss)]
    let mean: f64 = scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64;
    println!(
        "\nOverall: {} contracts, mean {:.2} (Grade {})",
        scores.len(),
        mean,
        provable_contracts::scoring::Grade::from_score(mean)
    );

    // Codebase scoring with binding
    let binding_path = Path::new("contracts/aprender/binding.yaml");
    if binding_path.exists() {
        let content = std::fs::read_to_string(binding_path).unwrap();
        let binding: provable_contracts::binding::BindingRegistry =
            serde_yaml::from_str(&content).unwrap();

        let mut parsed = Vec::new();
        for entry in &entries {
            let path = entry.path();
            let Ok(c) = provable_contracts::schema::parse_contract(&path) else {
                continue;
            };
            let stem = path.file_name().unwrap().to_str().unwrap().to_string();
            parsed.push((stem, c));
        }
        let refs: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();
        let codebase = provable_contracts::scoring::score_codebase(&refs, &binding);
        println!("\n=== Codebase Score (aprender) ===\n");
        print!("{codebase}");

        // Drift detection via git timestamps
        println!("\n=== Drift Detection ===\n");
        let bound_stems: std::collections::HashSet<&str> = binding
            .bindings
            .iter()
            .map(|b| b.contract.as_str())
            .collect();
        let stale = provable_contracts::scoring::drift::detect_stale_contracts(
            contracts_dir,
            binding_path,
            &bound_stems,
        );
        let drift =
            provable_contracts::scoring::drift::compute_drift(stale.len(), bound_stems.len());
        println!("  Stale contracts: {}/{}", stale.len(), bound_stems.len());
        println!("  Drift score: {drift:.2}");
        for s in &stale {
            println!("    - {s}");
        }

        // Pagerank-weighted gap analysis with drift
        println!("\n=== Full Codebase Score (pagerank + drift) ===\n");
        let index = provable_contracts::query::ContractIndex::from_directory(contracts_dir)
            .expect("contracts/ directory must exist");
        let pagerank: std::collections::HashMap<String, f64> = index
            .entries
            .iter()
            .filter_map(|e| index.cached_pagerank(&e.stem).map(|s| (e.stem.clone(), s)))
            .collect();
        let codebase_full = provable_contracts::scoring::score_codebase_full(
            &refs,
            &binding,
            Some(&pagerank),
            Some(drift),
        );
        print!("{codebase_full}");
    }
}
