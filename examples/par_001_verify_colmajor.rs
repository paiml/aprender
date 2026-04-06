//! PAR-001: Verify column-major Q6_K function works correctly
//!
//! This test compares the new fused_q6k_parallel_matvec function against
//! the manual transposed computation that was proven correct.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Verify Column-Major Q6_K Function ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let layer = &model.layers()[0];

    // Get real input
    let token_id: u32 = 26222;
    let hidden = model.embed(&[token_id]);
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + model.config().eps).sqrt();
    let normed: Vec<f32> = hidden
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| x / rms * w)
        .collect();

    println!("Input normed L2: {:.4}\n", l2_norm(&normed));

    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { v, .. } => {
            println!(
                "V: in_dim={}, out_dim={}, qtype={}",
                v.in_dim, v.out_dim, v.qtype
            );
            println!(
                "V data: {} bytes = {} superblocks",
                v.data.len(),
                v.data.len() / 210
            );

            // Method 1: Manual transposed computation (known correct from par_001_transpose_test)
            println!("\n=== Method 1: Manual transposed computation ===");
            let all_values = dequantize_q6_k(&v.data).expect("full dequant failed");
            println!("Dequantized {} values total", all_values.len());

            let mut transposed_output = vec![0.0f32; v.out_dim];
            for row in 0..v.out_dim {
                for col in 0..v.in_dim {
                    transposed_output[row] += all_values[col * 256 + row] * normed[col];
                }
            }
            println!("Transposed V output: L2={:.4}", l2_norm(&transposed_output));
            println!(
                "First 5: {:?}",
                &transposed_output[..5.min(transposed_output.len())]
            );

            // Method 2: Our new fused_q6k_parallel_matvec function
            println!("\n=== Method 2: fused_q6k_parallel_matvec function ===");
            let colmajor_output = fused_q6k_parallel_matvec(&v.data, &normed, v.in_dim, v.out_dim)
                .expect("colmajor matvec failed");
            println!("Column-major V output: L2={:.4}", l2_norm(&colmajor_output));
            println!(
                "First 5: {:?}",
                &colmajor_output[..5.min(colmajor_output.len())]
            );

            // Compare
            println!("\n=== Comparison ===");
            let diff: f32 = transposed_output
                .iter()
                .zip(colmajor_output.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            println!("L2 difference: {:.6}", diff);

            if diff < 1e-5 {
                println!("✓ Methods match!");
            } else {
                println!("✗ Methods differ!");

                // Find first divergence
                for (i, (a, b)) in transposed_output
                    .iter()
                    .zip(colmajor_output.iter())
                    .enumerate()
                {
                    if (a - b).abs() > 1e-4 {
                        println!("First divergence at index {}: {} vs {}", i, a, b);
                        break;
                    }
                }
            }
        },
        _ => println!("QKV is fused"),
    }

    println!("\n=== Verification complete ===");
}
