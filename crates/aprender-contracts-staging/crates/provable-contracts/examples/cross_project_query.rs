//! Example: Cross-project query with violations and coverage map.
//!
//! Demonstrates cross-project search features:
//! - `--call-sites`: Where contracts are used across sibling projects
//! - `--violations`: Binding gaps and unproven obligations
//! - `--coverage-map`: Per-project coverage matrix
//! - `--project <name>`: Filter to a single project
//! - `--all-projects`: Force cross-project scan
//! - `--rebuild-index`: Skip cache and rebuild
//!
//! Run from the workspace root:
//!   cargo run --example `cross_project_query`

use std::path::Path;

use provable_contracts::query::{self, ContractIndex, QueryParams};

fn main() {
    let contracts_dir = Path::new("contracts");
    let index =
        ContractIndex::from_directory(contracts_dir).expect("contracts/ directory must exist");

    // 1. Query with all cross-project enrichment
    println!("=== Cross-Project Query: \"softmax\" ===\n");
    let params = QueryParams {
        query: "softmax".to_string(),
        show_call_sites: true,
        show_violations: true,
        show_coverage_map: true,
        show_score: true,
        limit: 3,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // 2. Filter cross-project results to a single project
    println!("\n=== Filtered to aprender: \"softmax\" ===\n");
    let params = QueryParams {
        query: "softmax".to_string(),
        show_call_sites: true,
        show_violations: true,
        show_coverage_map: true,
        project_filter: Some("aprender".to_string()),
        limit: 2,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // 3. Force full cross-project scan with --all-projects
    println!("\n=== All Projects Scan: \"attention\" ===\n");
    let params = QueryParams {
        query: "attention".to_string(),
        all_projects: true,
        limit: 2,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // 4. JSON output for CI integration
    println!("\n=== JSON Output (first result) ===\n");
    let params = QueryParams {
        query: "attention".to_string(),
        show_call_sites: true,
        show_coverage_map: true,
        limit: 1,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    println!("{}", serde_json::to_string_pretty(&output).unwrap());

    // 5. Markdown output
    println!("\n=== Markdown Output ===\n");
    let params = QueryParams {
        query: "rmsnorm".to_string(),
        show_call_sites: true,
        show_violations: true,
        show_coverage_map: true,
        limit: 2,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{}", output.to_markdown());

    // 6. Tier filter: Foundation kernels only
    println!("\n=== Tier 1 (Foundation Kernels): \"kernel\" ===\n");
    let params = QueryParams {
        query: "kernel".to_string(),
        tier_filter: Some(1),
        limit: 5,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // 7. Class filter: Qwen equivalence class
    println!("\n=== Class E (Qwen): \"norm\" ===\n");
    let params = QueryParams {
        query: "norm".to_string(),
        class_filter: Some('E'),
        show_score: true,
        limit: 5,
        ..Default::default()
    };
    let output = query::execute(&index, &params);
    print!("{output}");

    // 8. Rebuild index from scratch (skip cache)
    println!("\n=== Rebuild Index ===\n");
    let fresh_index =
        ContractIndex::from_directory_opts(contracts_dir, true).expect("rebuild should succeed");
    println!("Rebuilt index with {} contracts", fresh_index.entries.len());
}
