//! PAR-001: Verify Q4_K fused matvec matches naive dequant+matmul
//!
//! This is the key test - if Q4_K fused matvec is wrong, everything is wrong

#![allow(clippy::needless_range_loop)]

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Verify Q4_K Fused Matvec ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    // Get Q projection weight (Q4_K)
    let layer = &model.layers()[0];
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &layer.qkv_weight else {
        panic!("Expected separate QKV")
    };

    println!(
        "Q weight: in={}, out={}, qtype={}",
        q_weight.in_dim, q_weight.out_dim, q_weight.qtype
    );
    println!("Q data bytes: {}", q_weight.data.len());

    // Get real input (normed embedding)
    let emb = model.embed(&[26222]); // "Once"
    let normed = rms_norm(&emb, &layer.attn_norm_weight, model.config().eps);
    println!("\nInput (normed embedding) L2: {:.4}", l2_norm(&normed));

    // Method 1: Fused Q4_K matvec
    println!("\n=== Method 1: fused_q4k_parallel_matvec ===");
    let fused_output =
        fused_q4k_parallel_matvec(&q_weight.data, &normed, q_weight.in_dim, q_weight.out_dim)
            .expect("fused matvec failed");
    println!("Fused output L2: {:.4}", l2_norm(&fused_output));
    println!(
        "First 10: {:?}",
        &fused_output[..10.min(fused_output.len())]
    );

    // Method 2: Naive dequantize + matmul
    println!("\n=== Method 2: Naive dequant + matmul ===");

    // Q4_K is row-major: row o has superblocks [o * (in_dim/256) .. (o+1) * (in_dim/256)]
    let all_weights = dequantize_q4_k(&q_weight.data).expect("dequant failed");
    println!("Dequantized {} weights", all_weights.len());

    // For row-major Q4_K: each row has in_dim/256 superblocks, each with 256 values
    let sb_per_row = q_weight.in_dim.div_ceil(256);
    println!("Super-blocks per row: {}", sb_per_row);

    let mut naive_output = vec![0.0f32; q_weight.out_dim];
    for o in 0..q_weight.out_dim {
        // Row o starts at superblock (o * sb_per_row)
        let sb_start = o * sb_per_row;
        let val_start = sb_start * 256;
        let mut dot = 0.0f32;
        for i in 0..q_weight.in_dim {
            let w_idx = val_start + i;
            if w_idx < all_weights.len() {
                dot += all_weights[w_idx] * normed[i];
            }
        }
        naive_output[o] = dot;
    }
    println!("Naive output L2: {:.4}", l2_norm(&naive_output));
    println!(
        "First 10: {:?}",
        &naive_output[..10.min(naive_output.len())]
    );

    // Compare
    println!("\n=== Comparison ===");
    let diff: f32 = fused_output
        .iter()
        .zip(naive_output.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("L2 difference: {:.6}", diff);

    // Element-wise differences
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for (i, (a, b)) in fused_output.iter().zip(naive_output.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            max_diff_idx = i;
        }
    }
    println!(
        "Max element-wise difference: {:.6} at index {}",
        max_diff, max_diff_idx
    );

    if diff < 1e-2 {
        println!("✓ Methods match well!");
    } else {
        println!("✗ Methods differ significantly!");

        // Show first few differences
        println!("\nFirst 20 comparisons:");
        for (i, (a, b)) in fused_output
            .iter()
            .zip(naive_output.iter())
            .enumerate()
            .take(20)
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

    println!("\n=== Complete ===");
}
