//! PAR-001: Verify V projection with real input
//!
//! Check if the V projection is computing correctly for the actual normed hidden state

#![allow(clippy::needless_range_loop)]

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Verify V Projection with Real Input ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let token_id: u32 = 26222; // "Once"
    println!(
        "Token: {} ('{}')",
        token_id,
        mapped
            .model
            .vocabulary()
            .expect("test")
            .get(token_id as usize)
            .map_or("?", |s| s.as_str())
    );

    // Get embedding and normalize
    let hidden = model.embed(&[token_id]);
    println!("\nEmbedding L2: {:.4}", l2_norm(&hidden));

    let layer = &model.layers()[0];
    let normed = rms_norm(&hidden, &layer.attn_norm_weight, model.config().eps);
    println!("Normed L2: {:.4}", l2_norm(&normed));

    // Get V weight
    let OwnedQKVWeights::Separate { v: v_weight, .. } = &layer.qkv_weight else {
        panic!("Expected separate QKV")
    };

    println!(
        "\nV weight: in={}, out={}, qtype={}",
        v_weight.in_dim, v_weight.out_dim, v_weight.qtype
    );
    println!("V data bytes: {}", v_weight.data.len());

    // Method 1: Column-major fused function
    println!("\n=== Method 1: fused_q6k_parallel_matvec ===");
    let fused_output =
        fused_q6k_parallel_matvec(&v_weight.data, &normed, v_weight.in_dim, v_weight.out_dim)
            .expect("fused failed");
    println!("Fused output L2: {:.4}", l2_norm(&fused_output));
    println!("First 8: {:?}", &fused_output[..8.min(fused_output.len())]);

    // Method 2: Naive dequantize then matmul (column-major)
    println!("\n=== Method 2: Naive dequant + column-major matmul ===");

    // For column-major Q6_K with out_dim=256:
    // - We have in_dim superblocks
    // - Each superblock has 256 values (one for each output row)
    // - Column c has values from superblock c

    let all_weights = dequantize_q6_k(&v_weight.data).expect("dequant failed");
    println!("Total dequantized values: {}", all_weights.len());
    println!(
        "Expected: in_dim * 256 = {} * 256 = {}",
        v_weight.in_dim,
        v_weight.in_dim * 256
    );

    // Column-major: column c (in_dim index c) is in superblock c
    // So weight[row, col] = all_weights[col * 256 + row]
    let mut naive_output = vec![0.0f32; v_weight.out_dim];
    for col in 0..v_weight.in_dim {
        let act = normed[col];
        for row in 0..v_weight.out_dim {
            let w_idx = col * 256 + row;
            if w_idx < all_weights.len() {
                naive_output[row] += all_weights[w_idx] * act;
            }
        }
    }
    println!("Naive output L2: {:.4}", l2_norm(&naive_output));
    println!("First 8: {:?}", &naive_output[..8.min(naive_output.len())]);

    // Compare
    println!("\n=== Comparison ===");
    let diff: f32 = fused_output
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
        for (i, (a, b)) in fused_output
            .iter()
            .zip(naive_output.iter())
            .enumerate()
            .take(10)
        {
            println!(
                "  {}: fused={:.6} vs naive={:.6} (diff={:.6})",
                i,
                a,
                b,
                (a - b).abs()
            );
        }
    }

    // Also check: what does dequantized weight look like?
    println!("\n=== Weight Statistics ===");
    let w_l2 = l2_norm(&all_weights);
    let w_min = all_weights.iter().copied().fold(f32::INFINITY, f32::min);
    let w_max = all_weights
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    println!(
        "Dequantized weights: L2={:.4}, min={:.4}, max={:.4}",
        w_l2, w_min, w_max
    );

    // Check first column (first 256 values)
    let col0 = &all_weights[..256];
    println!(
        "Column 0 (first 256): L2={:.4}, first 8: {:?}",
        l2_norm(col0),
        &col0[..8]
    );

    println!("\n=== Complete ===");
}
