//! Enforcement test: every contract YAML must produce a valid book page.

use provable_contracts::book_gen::generate_contract_page;
use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::parse_contract;
use std::path::Path;

#[test]
fn every_contract_generates_book_page() {
    let contract_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("contracts");

    let mut contracts = Vec::new();
    for entry in std::fs::read_dir(&contract_dir).expect("contracts/ directory must exist") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap()
                .to_string();
            let contract = parse_contract(&path)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            contracts.push((stem, contract));
        }
    }

    assert!(
        !contracts.is_empty(),
        "No contract YAML files found in {}",
        contract_dir.display()
    );

    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    for (stem, contract) in &contracts {
        let page = generate_contract_page(contract, stem, &graph);

        assert!(!page.is_empty(), "Book page for {stem} is empty");
        assert!(
            page.contains(&format!("# {stem}")),
            "Book page for {stem} missing title"
        );
        assert!(
            page.contains("## Equations"),
            "Book page for {stem} missing Equations section"
        );
        assert!(
            page.contains("$$") || page.contains("```\n"),
            "Book page for {stem} missing math or code block"
        );
    }
}
