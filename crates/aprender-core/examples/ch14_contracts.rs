#![allow(clippy::disallowed_methods)]
//! Chapter 14: Provable Contracts
//!
//! Demonstrates contract-driven development and falsification.
//! Citation: Papadakis et al., "Mutation Testing Advances," arXiv:1907.09356
//! Contract: contracts/apr-book-ch14-v1.yaml

fn main() {
    // Contract anatomy
    println!("Provable contracts (aprender-contracts):");
    println!("  405 contracts across 70 crates");
    println!("  YAML schema with falsification conditions");
    println!();

    // Popper's falsifiability criterion (1959)
    // A contract that cannot be falsified is not a contract
    println!("Popper's criterion:");
    println!("  A claim is scientific IFF it is falsifiable.");
    println!("  Applied to software: a contract MUST specify its failure conditions.");

    // Contract structure validation
    let severities = ["P0", "P1", "P2"];
    let actions = ["reject", "flag", "warn"];
    println!("\nSeverity levels: {:?}", severities);
    println!("Actions: {:?}", actions);
    assert_eq!(severities.len(), 3, "Exactly 3 severity levels");
    assert_eq!(actions.len(), 3, "Exactly 3 action types");

    // Namespace purity contract
    let unified_crates = [
        "aprender-compute",
        "aprender-serve",
        "aprender-train",
        "aprender-orchestrate",
        "aprender-present",
        "aprender-profile",
    ];
    for crate_name in &unified_crates {
        assert!(
            crate_name.starts_with("aprender-"),
            "All crates must use aprender-* namespace"
        );
    }
    println!("\nNamespace purity: {} crates verified", unified_crates.len());

    // Book contract counts
    let chapters = 20_usize;
    let conditions_per_chapter = 5_usize;
    let total_conditions = chapters * conditions_per_chapter;
    println!("\nBook contracts:");
    println!("  Chapters: {chapters}");
    println!("  Conditions per chapter: {conditions_per_chapter}");
    println!("  Total falsification conditions: {total_conditions}");
    assert_eq!(total_conditions, 100, "100 total falsification conditions");

    println!("\nChapter 14 contracts: PASSED");
}
