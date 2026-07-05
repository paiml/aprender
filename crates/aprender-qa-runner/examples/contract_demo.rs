//! Format Contract Demo (GH-190/191)
//!
//! Demonstrates the shared format contract system that defines behavioral
//! invariants between writer (aprender) and reader (realizar). The contract
//! is the single source of truth for tensor naming, dtype-byte mappings,
//! tolerances, and invariant definitions.
//!
//! Run with:
//! ```bash
//! cargo run --example contract_demo -p apr-qa-runner
//! ```

#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::too_many_lines)]

use aprender_qa_runner::{
    load_format_contract, lookup_tolerance, validate_dtype_bytes, validate_tensor_name,
    ContractTestConfig, InvariantId,
};

/// Run the format contract demo showing YAML contract validation features
fn main() {
    println!("=== Format Contract Demo (GH-190/191) ===\n");

    // =========================================================================
    // Load the embedded YAML contract
    // =========================================================================
    println!("=== Loading Contract ===\n");

    let contract = load_format_contract().expect("load format contract");
    println!("Version:     {}", contract.version);
    println!("Invariants:  {}", contract.invariants.len());
    println!("Dtypes:      {}", contract.dtype_bytes.mappings.len());
    println!("Tolerances:  {}", contract.tolerances.len());
    println!();

    // =========================================================================
    // Validate dtype-byte mappings (no duplicates)
    // =========================================================================
    println!("=== Dtype-Byte Validation ===\n");

    match validate_dtype_bytes(&contract) {
        Ok(()) => println!("[PASS] No duplicate GGML byte values"),
        Err(e) => println!("[FAIL] {e}"),
    }

    println!("\nGGML type mappings:");
    println!("  {:<8} Byte", "Dtype");
    println!("  {:<8} ----", "-----");
    for entry in &contract.dtype_bytes.mappings {
        println!("  {:<8} {}", entry.dtype, entry.byte);
    }
    println!();

    // =========================================================================
    // Tensor naming validation
    // =========================================================================
    println!("=== Tensor Naming Convention ===\n");

    println!("Convention: {}", contract.tensor_naming.convention);
    println!("Pattern:    {}\n", contract.tensor_naming.pattern);

    println!("Naming examples (canonical vs. forbidden):");
    for ex in &contract.tensor_naming.examples {
        println!("  {} (not: {})", ex.canonical, ex.forbidden);
    }
    println!();

    // Valid GGUF-short names
    let valid_names = [
        "0.q_proj.weight",
        "0.down_proj.weight",
        "23.v_proj.weight",
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    ];

    println!("Tensor name validation:");
    for name in &valid_names {
        let ok = validate_tensor_name(name, &contract);
        println!("  {:<30} {}", name, if ok { "VALID" } else { "INVALID" });
    }

    // Invalid HuggingFace-style names
    let invalid_names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ];

    for name in &invalid_names {
        let ok = validate_tensor_name(name, &contract);
        println!("  {:<30} {}", name, if ok { "VALID" } else { "INVALID" });
    }
    println!();

    // =========================================================================
    // Tolerance lookup
    // =========================================================================
    println!("=== Per-Dtype Tolerances ===\n");

    let dtype_hdr = "Dtype";
    let atol_hdr = "atol";
    let rtol_hdr = "rtol";
    println!("  {dtype_hdr:<8} {atol_hdr:>8} {rtol_hdr:>8}");
    let dash5 = "-----";
    let dash4 = "----";
    println!("  {dash5:<8} {dash4:>8} {dash4:>8}");
    for tol in &contract.tolerances {
        println!("  {:<8} {:>8} {:>8}", tol.dtype, tol.atol, tol.rtol);
    }
    println!();

    // Programmatic lookup
    let dtypes_to_check = ["F32", "F16", "Q4_K", "Q2_K", "UNKNOWN"];
    println!("Tolerance lookup:");
    for dtype in &dtypes_to_check {
        match lookup_tolerance(dtype, &contract) {
            Some((atol, rtol)) => {
                println!("  {dtype:<8} atol={atol}, rtol={rtol}");
            }
            None => {
                println!("  {dtype:<8} (not in contract)");
            }
        }
    }
    println!();

    // =========================================================================
    // Invariant definitions
    // =========================================================================
    println!("=== Contract Invariants ===\n");

    println!(
        "| {:<5} | {:<25} | {:<10} | {:<18} |",
        "ID", "Name", "Status", "Gate ID"
    );
    println!("|-------|---------------------------|------------|--------------------| ");
    for inv in &contract.invariants {
        let status = if inv.implemented {
            "Implemented"
        } else {
            "Contract"
        };
        println!(
            "| {:<5} | {:<25} | {:<10} | {:<18} |",
            inv.id, inv.name, status, inv.gate_id
        );
    }
    println!();

    for inv in &contract.invariants {
        println!("{}: {}", inv.id, inv.description);
        if !inv.catches.is_empty() {
            println!("  Catches: {}", inv.catches.join(", "));
        }
        if let Some(ref test) = inv.test {
            println!("  Test:    {test}");
        }
    }
    println!();

    // =========================================================================
    // InvariantId type-safe dispatch
    // =========================================================================
    println!("=== InvariantId Dispatch ===\n");

    let labels = ["I-1", "I-2", "I-3", "I-4", "I-5", "I-99"];
    for label in &labels {
        match InvariantId::from_label(label) {
            Some(id) => println!("  {label} -> gate_id: {}", id.gate_id()),
            None => println!("  {label} -> (unknown invariant)"),
        }
    }
    println!();

    // =========================================================================
    // ContractTestConfig
    // =========================================================================
    println!("=== ContractTestConfig ===\n");

    let default_config = ContractTestConfig::default();
    println!("Default invariants: {:?}", default_config.invariants);

    let custom_config = ContractTestConfig {
        invariants: vec!["I-2".to_string(), "I-3".to_string()],
    };
    println!("Custom invariants:  {:?}", custom_config.invariants);
    println!();

    // =========================================================================
    // Playbook YAML integration
    // =========================================================================
    println!("=== Playbook Integration ===\n");

    println!("Add to your playbook YAML:\n");
    println!("  contract_tests:");
    println!("    invariants: [\"I-2\", \"I-3\", \"I-4\", \"I-5\"]\n");
    println!("This enables contract invariant testing during certification.");
    println!("I-1 (Round-trip Identity) runs separately as the Golden Rule Test.\n");

    println!("=== Demo Complete ===");
}
