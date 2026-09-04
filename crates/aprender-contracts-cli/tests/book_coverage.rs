//! Enforcement test: every contract YAML must produce a valid book page.

use provable_contracts::book_gen::generate_contract_page;
use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{is_contract_yaml, parse_contract};
use std::path::Path;

#[test]
fn every_contract_generates_book_page() {
    let contract_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ is the parent of this crate")
        .parent()
        .expect("the repo root is the parent of crates/")
        .join("contracts");

    let mut contracts = Vec::new();
    for entry in std::fs::read_dir(&contract_dir).expect("contracts/ directory must exist") {
        let entry = entry.expect("the directory is readable");
        let path = entry.path();
        // `is_contract_yaml` is the single source of truth for "is this file a
        // contract" — the same predicate `pv lint`'s walker and the
        // `validate_contracts` integration test use. This loop used to apply
        // its own `.yaml` extension test, which made it a FOURTH walker with a
        // FOURTH answer: it parsed `contracts/binding.yaml`, a pv binding
        // registry with no `metadata:` block, and died on it.
        if is_contract_yaml(&path) {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("a .yaml path has a UTF-8 stem")
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
        // An "## Equations" section is demanded of a contract that HAS
        // equations, not of every contract. `metadata.kind` split the corpus
        // into kernels (which carry equations) and registries / patterns /
        // schemas / model-families (which do not, by design), and this
        // assertion predates that split: `contracts/apr-antigravity-parity-v1`
        // is a `kind: pattern` contract with no `equations:` block, and the
        // page generated for it is correct. Asserting the section
        // unconditionally does not catch a missing equation — it demands one
        // from contracts the schema exempts.
        if contract.equations.is_empty() {
            assert!(
                !page.contains("## Equations"),
                "Book page for {stem} claims an Equations section it has no equations for"
            );
        } else {
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
}
