//! PAR-001: Verify Q4_K fused dot vs naive for single row
//!
//! Test fused_q4k_dot directly with a single row to isolate the issue

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, fused_q4k_dot};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Verify Q4_K Fused Dot (Single Row) ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    // Get Q projection weight (Q4_K)
    let layer = &model.layers()[0];
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &layer.qkv_weight else {
        panic!("Expected separate QKV")
    };

    let in_dim = q_weight.in_dim;
    let out_dim = q_weight.out_dim;
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;

    println!("Q weight: in={}, out={}", in_dim, out_dim);
    println!("Super-blocks per row: {}", super_blocks_per_row);
    println!("Bytes per row: {}", bytes_per_row);
    println!("Total bytes: {}", q_weight.data.len());

    // Get real input (normed embedding)
    let emb = model.embed(&[26222]); // "Once"
    let normed = rms_norm(&emb, &layer.attn_norm_weight, model.config().eps);
    println!("\nInput L2: {:.4}", l2_norm(&normed));

    // Test row 0
    println!("\n=== Row 0 ===");
    let row0_start = 0;
    let row0_end = bytes_per_row;
    let row0_data = &q_weight.data[row0_start..row0_end];

    // Method 1: Fused dot
    let fused_result = fused_q4k_dot(row0_data, &normed).expect("fused_q4k_dot failed");
    println!("Fused result: {:.6}", fused_result);

    // Method 2: Naive dequant + dot
    let row0_weights = dequantize_q4_k(row0_data).expect("dequant failed");
    println!("Dequantized {} weights for row 0", row0_weights.len());

    let naive_result: f32 = row0_weights
        .iter()
        .zip(normed.iter())
        .map(|(w, a)| w * a)
        .sum();
    println!("Naive result: {:.6}", naive_result);

    println!("\nDifference: {:.6}", (fused_result - naive_result).abs());

    // Check the first few weights
    println!("\nFirst 10 dequantized weights: {:?}", &row0_weights[..10]);
    println!("First 10 activations: {:?}", &normed[..10]);

    // Check individual products
    println!("\nFirst 10 products:");
    for i in 0..10 {
        let prod = row0_weights[i] * normed[i];
        println!(
            "  {}: {:.6} * {:.6} = {:.6}",
            i, row0_weights[i], normed[i], prod
        );
    }

    // Test row 1
    println!("\n=== Row 1 ===");
    let row1_start = bytes_per_row;
    let row1_end = 2 * bytes_per_row;
    let row1_data = &q_weight.data[row1_start..row1_end];

    let fused_result1 = fused_q4k_dot(row1_data, &normed).expect("fused_q4k_dot failed");
    println!("Fused result: {:.6}", fused_result1);

    let row1_weights = dequantize_q4_k(row1_data).expect("dequant failed");
    let naive_result1: f32 = row1_weights
        .iter()
        .zip(normed.iter())
        .map(|(w, a)| w * a)
        .sum();
    println!("Naive result: {:.6}", naive_result1);

    println!("\nDifference: {:.6}", (fused_result1 - naive_result1).abs());

    // Also check: does dequantize_q4_k on the full weight data give the same row 0 weights?
    println!("\n=== Verify full dequant row ordering ===");
    let all_weights = dequantize_q4_k(&q_weight.data).expect("dequant full failed");
    println!("Total dequantized weights: {}", all_weights.len());

    // Row 0 should be at indices 0..in_dim
    let row0_from_full = &all_weights[0..in_dim];
    println!("Row 0 from full (first 10): {:?}", &row0_from_full[..10]);

    // Compare with per-row dequant
    let match_count = row0_weights
        .iter()
        .zip(row0_from_full.iter())
        .filter(|&(a, b)| (*a - *b).abs() < 1e-6)
        .count();
    println!("Row 0 match count: {}/{}", match_count, in_dim);

    if match_count < in_dim {
        println!("First mismatch:");
        for (i, (a, b)) in row0_weights.iter().zip(row0_from_full.iter()).enumerate() {
            if (a - b).abs() >= 1e-6 {
                println!("  Index {}: per-row={:.6} vs full={:.6}", i, a, b);
                break;
            }
        }
    }

    println!("\n=== Complete ===");
}
