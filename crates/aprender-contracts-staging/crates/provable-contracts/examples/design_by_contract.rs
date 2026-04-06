//! Design by Contract walkthrough.
//!
//! Demonstrates the core provable-contracts workflow: parse a contract
//! YAML, inspect its equations and proof obligations, diff two contract
//! versions, and check binding status from a registry.
//!
//! Run from the workspace root:
//!   cargo run --example `design_by_contract`

use std::path::Path;

use provable_contracts::binding::parse_binding_str;
use provable_contracts::diff::diff_contracts;
use provable_contracts::query::{self, ContractIndex, QueryParams};
use provable_contracts::schema::parse_contract_str;
use provable_contracts::scoring;

fn main() {
    // --- 1. Parse a contract from YAML ---
    let yaml = include_str!("../../../contracts/softmax-kernel-v1.yaml");
    let contract = parse_contract_str(yaml).expect("valid contract YAML");

    println!("Contract: {}", contract.metadata.description);
    println!("Version:  {}", contract.metadata.version);
    println!();

    // --- 2. Inspect equations ---
    println!("Equations ({}):", contract.equations.len());
    for (name, eq) in &contract.equations {
        println!("  {name}: {}", eq.formula);
        if let Some(ref domain) = eq.domain {
            println!("    domain:   {domain}");
        }
        for inv in &eq.invariants {
            println!("    invariant: {inv}");
        }
    }
    println!();

    // --- 3. Inspect proof obligations ---
    println!("Proof obligations ({}):", contract.proof_obligations.len());
    for ob in &contract.proof_obligations {
        let tol = ob
            .tolerance
            .map_or_else(String::new, |t| format!(" (tol={t})"));
        println!("  [{}] {}{tol}", ob.obligation_type, ob.property);
    }
    println!();

    // --- 4. Diff two contract versions ---
    let v2_yaml = yaml.replace("version: \"1.0.0\"", "version: \"1.1.0\"");
    let v2 = parse_contract_str(&v2_yaml).expect("v2 contract");
    let diff = diff_contracts(&contract, &v2);
    println!(
        "Diff {} -> {}: suggested bump = {}",
        diff.old_version, diff.new_version, diff.suggested_bump
    );
    println!();

    // --- 5. Check binding status ---
    let binding_yaml = r#"
version: "1.0.0"
target_crate: aprender
bindings:
  - contract: softmax-kernel-v1.yaml
    equation: softmax
    module_path: "aprender::nn::softmax"
    function: softmax_f32
    status: implemented
"#;
    let registry = parse_binding_str(binding_yaml).expect("valid binding YAML");
    println!(
        "Binding registry: {} ({} bindings)",
        registry.target_crate,
        registry.bindings.len()
    );
    for b in &registry.bindings {
        println!("  {} :: {} -> {}", b.contract, b.equation, b.status);
    }
    println!();

    // --- 6. Score the contract ---
    let score = scoring::score_contract(&contract, Some(&registry), "softmax-kernel-v1");
    print!("{score}");
    println!();

    // --- 7. Query the contract index ---
    let contracts_dir = Path::new("contracts");
    if contracts_dir.exists() {
        let index =
            ContractIndex::from_directory(contracts_dir).expect("contracts/ directory must exist");
        let params = QueryParams {
            query: "softmax numerical stability".to_string(),
            show_score: true,
            show_pagerank: true,
            limit: 3,
            ..Default::default()
        };
        let output = query::execute(&index, &params);
        println!("--- Query: \"softmax numerical stability\" (top 3) ---\n");
        print!("{output}");

        // --- 8. Tier filter: Foundation kernels only ---
        println!("--- Tier 1 (Foundation Kernels) ---\n");
        let params = QueryParams {
            query: "kernel".to_string(),
            tier_filter: Some(1),
            limit: 5,
            ..Default::default()
        };
        let output = query::execute(&index, &params);
        print!("{output}");

        // --- 9. Class filter: Llama/Mistral equivalence class ---
        println!("--- Class A (Llama/Mistral) ---\n");
        let params = QueryParams {
            query: "attention".to_string(),
            class_filter: Some('A'),
            show_score: true,
            limit: 5,
            ..Default::default()
        };
        let output = query::execute(&index, &params);
        print!("{output}");
    }
}
