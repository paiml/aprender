#![allow(clippy::disallowed_methods)]
//! Design by Contract in Trueno
//!
//! Demonstrates kernel-level contracts: quantization format validation,
//! GEMV shape invariants, and expected byte calculations.
//!
//! Run with: `cargo run --example design_by_contract`

use trueno::contracts::{self, QuantFormat, TensorLayout, Q4_K, Q5_K, Q6_K, Q8_0, STACK_LAYOUT};

fn main() {
    println!("=== Trueno Design by Contract ===\n");

    // --- 1. Stack layout contract ---
    println!("1. Stack layout contract:");
    assert_eq!(STACK_LAYOUT, TensorLayout::RowMajor);
    println!("   STACK_LAYOUT = {:?} (the ONLY supported layout)", STACK_LAYOUT);

    // --- 2. Quantization format constants ---
    println!("\n2. Quantization format contracts:");
    print_format(&Q4_K);
    print_format(&Q5_K);
    print_format(&Q6_K);
    print_format(&Q8_0);

    // --- 3. Expected bytes calculation ---
    println!("\n3. Buffer size contracts (4096x4096 weight matrix):");
    let rows = 4096;
    let cols = 4096;
    println!("   Q4_K: {} bytes", Q4_K.expected_bytes(rows, cols));
    println!("   Q5_K: {} bytes", Q5_K.expected_bytes(rows, cols));
    println!("   Q6_K: {} bytes", Q6_K.expected_bytes(rows, cols));
    println!("   Q8_0: {} bytes", Q8_0.expected_bytes(rows, cols));
    println!("   F32:  {} bytes", rows * cols * 4);

    // --- 4. Buffer validation (correct size -- passes) ---
    println!("\n4. Buffer validation (correct size):");
    let expected = Q4_K.expected_bytes(rows, cols);
    match Q4_K.validate_buffer("attn_q.weight", expected, rows, cols) {
        Ok(()) => println!("   PASS: Q4_K buffer accepted ({expected} bytes)"),
        Err(e) => println!("   FAIL: {e}"),
    }

    // --- 5. Buffer validation (wrong size -- contract violation) ---
    println!("\n5. Buffer validation (wrong size -- contract violation):");
    match Q4_K.validate_buffer("attn_q.weight", 1000, rows, cols) {
        Ok(()) => println!("   UNEXPECTED PASS"),
        Err(e) => println!("   REJECTED (expected): {e}"),
    }

    // --- 6. validate_weight_buffer with GGML type ID ---
    println!("\n6. GGML type ID lookup + validation:");
    let buf_size = Q6_K.expected_bytes(rows, cols);
    match contracts::validate_weight_buffer("ffn_down.weight", 14, buf_size, rows, cols) {
        Ok(()) => println!("   PASS: GGML type 14 (Q6_K) validated"),
        Err(e) => println!("   FAIL: {e}"),
    }
    // Unknown GGML type
    match contracts::validate_weight_buffer("unknown.weight", 99, 1000, rows, cols) {
        Ok(()) => println!("   UNEXPECTED PASS"),
        Err(e) => println!("   REJECTED (expected): {e}"),
    }

    // --- 7. F32 buffer validation ---
    println!("\n7. F32 buffer validation:");
    match contracts::validate_f32_buffer("embed.weight", rows * cols, rows, cols) {
        Ok(()) => println!("   PASS: F32 buffer accepted ({} elements)", rows * cols),
        Err(e) => println!("   FAIL: {e}"),
    }
    match contracts::validate_f32_buffer("embed.weight", 100, rows, cols) {
        Ok(()) => println!("   UNEXPECTED PASS"),
        Err(e) => println!("   REJECTED (expected): {e}"),
    }

    // --- 8. GEMV shape validation ---
    println!("\n8. GEMV shape contract:");
    let out_dim = 4096;
    let in_dim = 4096;
    match contracts::validate_gemv_shapes("attn_o.weight", out_dim, in_dim, in_dim, out_dim) {
        Ok(()) => println!("   PASS: W[{out_dim},{in_dim}] x[{in_dim}] -> y[{out_dim}]"),
        Err(e) => println!("   FAIL: {e}"),
    }
    // Input dimension mismatch
    match contracts::validate_gemv_shapes("attn_o.weight", out_dim, in_dim, 999, out_dim) {
        Ok(()) => println!("   UNEXPECTED PASS"),
        Err(e) => println!("   REJECTED (expected): {e}"),
    }
    // Output dimension mismatch
    match contracts::validate_gemv_shapes("attn_o.weight", out_dim, in_dim, in_dim, 999) {
        Ok(()) => println!("   UNEXPECTED PASS"),
        Err(e) => println!("   REJECTED (expected): {e}"),
    }

    // --- 9. Non-aligned columns (edge case) ---
    println!("\n9. Non-aligned columns (Q4_K, block_size=256):");
    println!(
        "   100 cols: ceil(100/256)=1 block/row -> {} bytes for 10 rows",
        Q4_K.expected_bytes(10, 100)
    );
    println!(
        "   300 cols: ceil(300/256)=2 blocks/row -> {} bytes for 10 rows",
        Q4_K.expected_bytes(10, 300)
    );

    println!("\n=== All contract demonstrations complete ===");
}

fn print_format(fmt: &QuantFormat) {
    println!(
        "   {}: {} elements/block, {} bytes/block, GGML type {}",
        fmt.name, fmt.block_size, fmt.block_bytes, fmt.ggml_type_id
    );
}
