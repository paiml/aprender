//! Demonstrates scoring and validating infrastructure contracts (MCP, CLI, HTTP).
//!
//! These contracts cover non-kernel domains: protocol SDKs, CLI tools, and HTTP
//! clients. They focus on state machines, determinism, and safety invariants
//! rather than numerical accuracy.
//!
//! Run with:
//!   cargo run --example infrastructure_contracts

use std::path::Path;

use provable_contracts::schema::parse_contract;
use provable_contracts::scoring::score_contract;

fn main() {
    println!("Infrastructure Contract Validation & Scoring");
    println!("=============================================\n");

    let contract_dir = Path::new("contracts");

    let infra = [
        ("MCP Protocol SDK (pmcp)", "pmcp/mcp-protocol-sdk-v1.yaml"),
        ("HTTP Client (rurl)", "rurl/http-client-v1.yaml"),
        ("APR CLI (apr-cli)", "aprender/apr-cli-v1.yaml"),
    ];

    let mut total_eq = 0;
    let mut total_ob = 0;
    let mut total_ft = 0;
    let mut composites = Vec::new();

    for (label, rel) in &infra {
        let path = contract_dir.join(rel);
        if !path.exists() {
            println!("--- {label} ---\n  [SKIP] {rel} not found\n");
            continue;
        }

        println!("--- {label} ---\n");
        match parse_contract(&path) {
            Ok(contract) => {
                let eq_count = contract.equations.len();
                let ob_count = contract.proof_obligations.len();
                let ft_count = contract.falsification_tests.len();
                let kani_count = contract.kani_harnesses.len();

                println!("  Equations:           {eq_count}");
                println!("  Proof obligations:   {ob_count}");
                println!("  Falsification tests: {ft_count}");
                println!("  Kani harnesses:      {kani_count}");

                for (name, eq) in &contract.equations {
                    let pre = eq.preconditions.len();
                    let post = eq.postconditions.len();
                    let inv = eq.invariants.len();
                    println!("    {name}: {pre} pre, {post} post, {inv} inv");
                }

                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                let score = score_contract(&contract, None, stem);
                println!("\n  {score}");

                total_eq += eq_count;
                total_ob += ob_count;
                total_ft += ft_count;
                composites.push(score.composite);
            }
            Err(e) => println!("  ERROR: {e}"),
        }
    }

    if !composites.is_empty() {
        let avg = composites.iter().sum::<f64>() / composites.len() as f64;
        println!("=== Infrastructure Summary ===");
        println!("  Contracts:    {}", composites.len());
        println!("  Equations:    {total_eq}");
        println!("  Obligations:  {total_ob}");
        println!("  Falsif tests: {total_ft}");
        println!("  Mean score:   {avg:.2}");
    }
}
