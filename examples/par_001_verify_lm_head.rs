//! PAR-001: Verify Q6_K LM head (row-major) works correctly
//!
//! LM head has out_dim=32000, so it uses row-major Q6_K.
//! This test compares the row-major function with naive dequant+matmul.

#![allow(clippy::needless_range_loop)]

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Verify LM head (row-major Q6_K) ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let lm_head = &model.lm_head_weight();

    println!(
        "LM head: in_dim={}, out_dim={}, qtype={}",
        lm_head.in_dim, lm_head.out_dim, lm_head.qtype
    );
    println!("LM head data: {} bytes", lm_head.data.len());

    // Verify dimensions make sense
    let in_sb_count = lm_head.in_dim.div_ceil(256);
    let total_sb = lm_head.out_dim * in_sb_count;
    let expected_bytes = total_sb * 210; // Q6_K is 210 bytes per superblock
    println!("\nExpected layout:");
    println!("  in_dim super-blocks per row: {}", in_sb_count);
    println!(
        "  total super-blocks: {} x {} = {}",
        lm_head.out_dim, in_sb_count, total_sb
    );
    println!("  expected bytes: {} x 210 = {}", total_sb, expected_bytes);
    println!("  actual bytes: {}", lm_head.data.len());
    println!("  match: {}", expected_bytes == lm_head.data.len());

    // Create random-ish input (simulating normalized hidden state)
    let input: Vec<f32> = (0..lm_head.in_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    println!("\nInput L2: {:.4}", l2_norm(&input));

    // Method 1: Row-major fused function
    println!("\n=== Method 1: fused_q6k_parallel_matvec (row-major) ===");
    let rowmajor_output =
        fused_q6k_parallel_matvec(&lm_head.data, &input, lm_head.in_dim, lm_head.out_dim)
            .expect("row-major matvec failed");
    println!("Row-major output L2: {:.4}", l2_norm(&rowmajor_output));
    println!(
        "First 10: {:?}",
        &rowmajor_output[..10.min(rowmajor_output.len())]
    );

    // Method 2: Naive dequantize then matmul
    println!("\n=== Method 2: Naive dequant + matmul ===");
    let all_weights = dequantize_q6_k(&lm_head.data).expect("dequant failed");
    println!("Dequantized {} weights", all_weights.len());
    println!(
        "Expected weights: {} x {} = {}",
        lm_head.out_dim,
        lm_head.in_dim,
        lm_head.out_dim * lm_head.in_dim
    );

    // Row-major: row o uses weights [o * in_dim .. (o+1) * in_dim]
    // But Q6_K has 256 values per superblock. Each row has (in_dim/256) superblocks.
    println!("Super-blocks per row: {}", in_sb_count);

    // Only compute first 100 outputs for speed (32000 takes too long)
    let check_count = 100.min(lm_head.out_dim);
    let mut naive_output = vec![0.0f32; check_count];
    for o in 0..check_count {
        // Row o's weights start at super-block (o * in_sb_count)
        let sb_start = o * in_sb_count;
        let val_start = sb_start * 256;
        let mut dot = 0.0f32;
        for i in 0..lm_head.in_dim {
            let w_idx = val_start + i;
            if w_idx < all_weights.len() {
                dot += all_weights[w_idx] * input[i];
            }
        }
        naive_output[o] = dot;
    }
    println!(
        "Naive output (first {}) L2: {:.4}",
        check_count,
        l2_norm(&naive_output)
    );
    println!(
        "First 10: {:?}",
        &naive_output[..10.min(naive_output.len())]
    );

    // Compare first `check_count` outputs
    println!("\n=== Comparison (first {} outputs) ===", check_count);
    let diff: f32 = rowmajor_output[..check_count]
        .iter()
        .zip(naive_output.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("L2 difference: {:.6}", diff);

    if diff < 1e-3 {
        println!("✓ Methods match!");
    } else {
        println!("✗ Methods differ!");

        // Find first divergence
        for (i, (a, b)) in rowmajor_output.iter().zip(naive_output.iter()).enumerate() {
            if (a - b).abs() > 1e-3 {
                println!(
                    "First divergence at index {}: fused={:.6} vs naive={:.6}",
                    i, a, b
                );
                break;
            }
        }
    }

    // Also check some specific high token IDs to see if there's a pattern
    println!("\n=== Checking specific output indices ===");
    let check_indices = [0, 1, 100, 1000, 10000, 31999];
    for &idx in &check_indices {
        if idx < lm_head.out_dim {
            // Compute naive for this specific index
            let sb_start = idx * in_sb_count;
            let val_start = sb_start * 256;
            let mut naive_val = 0.0f32;
            for i in 0..lm_head.in_dim {
                let w_idx = val_start + i;
                if w_idx < all_weights.len() {
                    naive_val += all_weights[w_idx] * input[i];
                }
            }
            println!(
                "  idx {}: fused={:.6}, naive={:.6}, diff={:.6}",
                idx,
                rowmajor_output[idx],
                naive_val,
                (rowmajor_output[idx] - naive_val).abs()
            );
        }
    }

    println!("\n=== Verification complete ===");
}
