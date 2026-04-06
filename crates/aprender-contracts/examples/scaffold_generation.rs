//! Demonstrates generating Rust trait scaffolds from contracts.
//!
//! This example shows how to:
//! 1. Generate a generic `KernelContract` trait from a parsed contract
//! 2. Generate a named standalone trait (e.g., `SoftmaxKernelV1`) for a specific contract
//! 3. Generate failing `#[test]` stubs from falsification tests
//!
//! Run with:
//!   cargo run --example scaffold_generation
//!
//! Or generate from a specific contract file:
//!   cargo run --example scaffold_generation -- contracts/softmax-kernel-v1.yaml

use std::path::{Path, PathBuf};

use provable_contracts::scaffold::{
    generate_contract_tests, generate_standalone_trait, generate_trait,
};
use provable_contracts::schema::{parse_contract, parse_contract_str};

/// Inline contract with multiple equations demonstrating richer scaffold output.
const ATTENTION_CONTRACT: &str = r#"
metadata:
  version: "1.0.0"
  description: "Scaled dot-product attention kernel"
  references:
    - "Vaswani et al. (2017) Attention Is All You Need"
equations:
  scaled_dot_product:
    formula: "Attention(Q,K,V) = softmax(QK^T / √d_k) V"
    domain: "Q ∈ ℝ^{n×d_k}, K ∈ ℝ^{m×d_k}, V ∈ ℝ^{m×d_v}"
    codomain: "Attention ∈ ℝ^{n×d_v}"
    invariants:
      - "attention weights sum to 1.0 per row"
      - "output dimensions match (n, d_v)"
  causal_mask:
    formula: "mask(i,j) = -∞ if j > i, else 0"
    domain: "i ∈ [0,n), j ∈ [0,m)"
    invariants:
      - "upper triangle is -∞"
      - "lower triangle + diagonal is 0"
proof_obligations:
  - type: invariant
    property: "Attention weights form valid distribution"
    formal: "∀ row i: Σ_j softmax(scores)_{i,j} = 1.0"
    tolerance: 1.0e-6
  - type: equivalence
    property: "SIMD matches scalar attention"
    tolerance: 8.0
    applies_to: simd
  - type: bound
    property: "Output dimensions correct"
    formal: "output.shape = (n, d_v)"
falsification_tests:
  - id: FALSIFY-ATT-001
    rule: "Attention weights form valid distribution"
    prediction: "Each row of attention weights sums to 1.0"
    if_fails: "Softmax not applied row-wise or scaling factor wrong"
  - id: FALSIFY-ATT-002
    rule: "SIMD matches scalar attention"
    prediction: "max |simd - scalar| <= 8 ULP"
    if_fails: "FMA ordering differs between scalar and SIMD paths"
  - id: FALSIFY-ATT-003
    rule: "Output dimensions correct"
    prediction: "output.shape == (n, d_v)"
    if_fails: "Transpose error in V multiplication"
kani_harnesses:
  - id: KANI-ATT-001
    obligation: ATT-INV-001
    property: "Row normalization"
    bound: 8
    strategy: stub_float
"#;

fn main() {
    println!("=== provable-contracts: Scaffold Generation Example ===\n");

    // --- Part 1: Generate a generic KernelContract trait ---
    println!("--- Part 1: Generic KernelContract trait ---\n");

    let contract = parse_contract_str(ATTENTION_CONTRACT).expect("inline contract should parse");

    let trait_code = generate_trait(&contract);
    println!("{trait_code}");

    // --- Part 2: Generate a named standalone trait ---
    println!("--- Part 2: Standalone named trait (AttentionKernelV1) ---\n");

    let standalone = generate_standalone_trait(&contract, "attention-kernel-v1");
    println!("{standalone}");

    // --- Part 3: Generate failing test stubs ---
    println!("--- Part 3: Failing test stubs from falsification tests ---\n");

    let tests = generate_contract_tests(&contract);
    println!("{tests}");

    // --- Part 4: Summary statistics ---
    println!("--- Part 4: Scaffold summary ---\n");
    println!(
        "  Contract: \"{}\" v{}",
        contract.metadata.description, contract.metadata.version
    );
    println!("  Equations:           {}", contract.equations.len());
    println!(
        "  Proof obligations:   {}",
        contract.proof_obligations.len()
    );
    println!(
        "  Falsification tests: {} (generates {} test stubs)",
        contract.falsification_tests.len(),
        contract.falsification_tests.len()
    );
    println!("  Kani harnesses:      {}", contract.kani_harnesses.len());

    // Count the generated trait methods
    let method_count = contract.equations.len();
    println!("  Generated methods:   {method_count}");

    // --- Part 5: Generate from a file if provided ---
    println!("\n--- Part 5: Generate from contract file ---\n");

    if let Some(path_arg) = std::env::args().nth(1) {
        generate_from_file(&PathBuf::from(path_arg));
    } else {
        // Try a well-known contract from the workspace
        let default_path = Path::new("contracts/softmax-kernel-v1.yaml");
        if default_path.exists() {
            generate_from_file(default_path);
        } else {
            println!("  No CLI argument and contracts/softmax-kernel-v1.yaml not found.");
            println!("  Pass a YAML path to generate scaffolds from a real contract.");
        }
    }
}

/// Load a contract file and generate all scaffold outputs.
fn generate_from_file(path: &Path) {
    println!("  Source: {}\n", path.display());

    let contract = match parse_contract(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Failed to parse {}: {e}", path.display());
            return;
        }
    };

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Standalone trait for this specific contract
    println!("  // --- Standalone trait for {stem} ---\n");
    let standalone = generate_standalone_trait(&contract, stem);
    println!("{standalone}");

    // Test stubs
    println!("  // --- Test stubs ---\n");
    let tests = generate_contract_tests(&contract);
    println!("{tests}");

    println!(
        "  Generated {} method(s) and {} test stub(s) from {stem}.",
        contract.equations.len(),
        contract.falsification_tests.len()
    );
}
