//! PAR-001h: Compare fused Q6_K dot vs naive dequantize+dot
//!
//! The dequantization produces reasonable values but the projection output
//! is still wrong. Let's verify the fused dot matches naive dequantize+dot.

#![allow(clippy::needless_range_loop)]

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_dot, fused_q6k_parallel_matvec};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001h: Fused vs Naive Q6_K Dot Product ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[0];
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { v, .. } => {
            println!("V weight: in_dim={}, out_dim={}", v.in_dim, v.out_dim);

            let bytes_per_row = (v.in_dim / 256) * 210;
            println!("bytes_per_row: {}\n", bytes_per_row);

            // Create test input
            let input: Vec<f32> = (0..v.in_dim).map(|i| (i as f32 * 0.001).sin()).collect();
            let input_l2 = (input.iter().map(|x| x * x).sum::<f32>()).sqrt();
            println!("Input L2: {:.4}\n", input_l2);

            // Test row 0
            println!("=== Row 0 Comparison ===");
            let row0_data = &v.data[0..bytes_per_row];

            // Naive: dequantize then dot
            let row0_weights = dequantize_q6_k(row0_data).expect("dequant failed");
            let naive_dot: f32 = row0_weights
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum();
            println!("Naive (dequant + dot): {:.6}", naive_dot);

            // Fused: direct dot on quantized data
            let fused_dot = fused_q6k_dot(row0_data, &input).expect("fused failed");
            println!("Fused dot: {:.6}", fused_dot);

            let diff = (naive_dot - fused_dot).abs();
            let rel_diff = diff / naive_dot.abs().max(1e-10);
            println!(
                "Difference: {:.6} (relative: {:.6}%)",
                diff,
                rel_diff * 100.0
            );

            // Test full matvec
            println!("\n=== Full Matvec Comparison ===");

            // Naive: dequantize all rows then matmul
            let mut naive_output = vec![0.0f32; v.out_dim];
            for row in 0..v.out_dim {
                let row_start = row * bytes_per_row;
                let row_data = &v.data[row_start..row_start + bytes_per_row];
                let row_weights = dequantize_q6_k(row_data).expect("dequant failed");
                naive_output[row] = row_weights
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum();
            }
            let naive_l2 = (naive_output.iter().map(|x| x * x).sum::<f32>()).sqrt();

            // Fused: parallel matvec
            let fused_output = fused_q6k_parallel_matvec(&v.data, &input, v.in_dim, v.out_dim)
                .expect("fused matvec failed");
            let fused_l2 = (fused_output.iter().map(|x| x * x).sum::<f32>()).sqrt();

            println!("Naive output L2: {:.6}", naive_l2);
            println!("Fused output L2: {:.6}", fused_l2);

            // Element-wise comparison
            let mut max_diff = 0.0f32;
            let mut max_diff_idx = 0;
            let mut sum_sq_diff = 0.0f32;
            for i in 0..v.out_dim {
                let diff = (naive_output[i] - fused_output[i]).abs();
                sum_sq_diff += diff * diff;
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_idx = i;
                }
            }
            let rmse = (sum_sq_diff / v.out_dim as f32).sqrt();
            println!(
                "\nMax difference: {:.6} at index {}",
                max_diff, max_diff_idx
            );
            println!("RMSE: {:.6}", rmse);

            // Show first 5 outputs
            println!("\nFirst 5 outputs:");
            for i in 0..5 {
                println!(
                    "  [{}] naive={:.6}, fused={:.6}, diff={:.6}",
                    i,
                    naive_output[i],
                    fused_output[i],
                    (naive_output[i] - fused_output[i]).abs()
                );
            }

            // Now test with actual model input (the normed hidden state)
            println!("\n\n=== Test with Real Model Input ===");
            let token_id: u32 = 26222; // "Once"
            let hidden = model.embed(&[token_id]);

            // Apply RMS norm
            let hidden_dim = model.config().hidden_dim;
            let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + model.config().eps).sqrt();
            let normed: Vec<f32> = hidden
                .iter()
                .zip(layer.attn_norm_weight.iter())
                .map(|(&x, &w)| x / rms * w)
                .collect();
            let normed_l2 = (normed.iter().map(|x| x * x).sum::<f32>()).sqrt();
            println!("Normed input L2: {:.4}", normed_l2);

            // Compute V projection with naive and fused
            let mut naive_v = vec![0.0f32; v.out_dim];
            for row in 0..v.out_dim {
                let row_start = row * bytes_per_row;
                let row_data = &v.data[row_start..row_start + bytes_per_row];
                let row_weights = dequantize_q6_k(row_data).expect("dequant failed");
                naive_v[row] = row_weights
                    .iter()
                    .zip(normed.iter())
                    .map(|(w, x)| w * x)
                    .sum();
            }
            let naive_v_l2 = (naive_v.iter().map(|x| x * x).sum::<f32>()).sqrt();

            let fused_v = fused_q6k_parallel_matvec(&v.data, &normed, v.in_dim, v.out_dim)
                .expect("fused matvec failed");
            let fused_v_l2 = (fused_v.iter().map(|x| x * x).sum::<f32>()).sqrt();

            println!("Naive V output L2: {:.6}", naive_v_l2);
            println!("Fused V output L2: {:.6}", fused_v_l2);

            let naive_nonzero = naive_v.iter().filter(|&&x| x.abs() > 0.01).count();
            let fused_nonzero = fused_v.iter().filter(|&&x| x.abs() > 0.01).count();
            println!("Naive V non-zero (>0.01): {}/{}", naive_nonzero, v.out_dim);
            println!("Fused V non-zero (>0.01): {}/{}", fused_nonzero, v.out_dim);
        },
        _ => println!("QKV is fused"),
    }

    println!("\n=== Analysis complete ===");
}
