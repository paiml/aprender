//! PAR-001: Verify Q6_K ffn_down (row-major) works correctly
//!
//! ffn_down has out_dim=2048, so it uses row-major Q6_K.
//! This test compares the row-major function with naive dequant+matmul.

#![allow(clippy::needless_range_loop)]

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Verify ffn_down (row-major Q6_K) ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[0];
    let ffn_down = &layer.ffn_down_weight;

    println!(
        "ffn_down: in_dim={}, out_dim={}, qtype={}",
        ffn_down.in_dim, ffn_down.out_dim, ffn_down.qtype
    );
    println!("ffn_down data: {} bytes", ffn_down.data.len());

    // Create random-ish input
    let input: Vec<f32> = (0..ffn_down.in_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    println!("\nInput L2: {:.4}", l2_norm(&input));

    // Method 1: Row-major fused function
    println!("\n=== Method 1: fused_q6k_parallel_matvec (row-major) ===");
    let rowmajor_output =
        fused_q6k_parallel_matvec(&ffn_down.data, &input, ffn_down.in_dim, ffn_down.out_dim)
            .expect("row-major matvec failed");
    println!("Row-major output L2: {:.4}", l2_norm(&rowmajor_output));
    println!(
        "First 5: {:?}",
        &rowmajor_output[..5.min(rowmajor_output.len())]
    );

    // Method 2: Naive dequantize then matmul
    println!("\n=== Method 2: Naive dequant + matmul ===");
    let all_weights = dequantize_q6_k(&ffn_down.data).expect("dequant failed");
    println!("Dequantized {} weights", all_weights.len());

    // Row-major: row o uses weights [o * in_dim .. (o+1) * in_dim]
    // But Q6_K has 256 values per superblock. Each row has (in_dim/256) superblocks.
    let in_sb_count = ffn_down.in_dim.div_ceil(256);
    println!("Super-blocks per row: {}", in_sb_count);

    let mut naive_output = vec![0.0f32; ffn_down.out_dim];
    for o in 0..ffn_down.out_dim {
        // Row o's weights start at super-block (o * in_sb_count)
        let sb_start = o * in_sb_count;
        let val_start = sb_start * 256;
        let mut dot = 0.0f32;
        for i in 0..ffn_down.in_dim {
            let w_idx = val_start + i;
            if w_idx < all_weights.len() {
                dot += all_weights[w_idx] * input[i];
            }
        }
        naive_output[o] = dot;
    }
    println!("Naive output L2: {:.4}", l2_norm(&naive_output));
    println!("First 5: {:?}", &naive_output[..5.min(naive_output.len())]);

    // Compare
    println!("\n=== Comparison ===");
    let diff: f32 = rowmajor_output
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

    println!("\n=== Verification complete ===");
}
