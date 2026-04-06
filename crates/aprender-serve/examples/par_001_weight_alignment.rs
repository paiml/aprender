//! PAR-001j: Check weight-input alignment across all rows
//!
//! V row 0 has very low alignment with input. Let's check if this
//! is consistent across all V rows and compare to K (same shape, different qtype).

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k_simd, dequantize_q6_k};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001j: Weight-Input Alignment Analysis ===\n");

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
    let normed_l2 = l2_norm(&normed);

    println!("Input normed L2: {:.4}\n", normed_l2);

    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q: _, k, v } => {
            let k_bytes_per_row = (k.in_dim / 256) * 144;
            let v_bytes_per_row = (v.in_dim / 256) * 210;

            println!("Checking K (Q4_K) alignment for first 10 rows:");
            let mut k_dots = Vec::with_capacity(k.out_dim);
            for row in 0..k.out_dim.min(10) {
                let row_data = &k.data[row * k_bytes_per_row..(row + 1) * k_bytes_per_row];
                let row_weights = dequantize_q4_k_simd(row_data).expect("K dequant failed");
                let dot: f32 = row_weights
                    .iter()
                    .zip(normed.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                let cos = dot / (l2_norm(&row_weights) * normed_l2);
                k_dots.push(dot);
                println!(
                    "  K row {:3}: dot={:+.6}, cos={:+.6}, L2={:.4}",
                    row,
                    dot,
                    cos,
                    l2_norm(&row_weights)
                );
            }

            println!("\nChecking V (Q6_K) alignment for first 10 rows:");
            let mut v_dots = Vec::with_capacity(v.out_dim);
            for row in 0..v.out_dim.min(10) {
                let row_data = &v.data[row * v_bytes_per_row..(row + 1) * v_bytes_per_row];
                let row_weights = dequantize_q6_k(row_data).expect("V dequant failed");
                let dot: f32 = row_weights
                    .iter()
                    .zip(normed.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                let cos = dot / (l2_norm(&row_weights) * normed_l2);
                v_dots.push(dot);
                println!(
                    "  V row {:3}: dot={:+.6}, cos={:+.6}, L2={:.4}",
                    row,
                    dot,
                    cos,
                    l2_norm(&row_weights)
                );
            }

            // Summary statistics
            println!("\n=== Summary Statistics (all rows) ===");

            // Compute all K dots
            let mut k_all_dots = Vec::with_capacity(k.out_dim);
            for row in 0..k.out_dim {
                let row_data = &k.data[row * k_bytes_per_row..(row + 1) * k_bytes_per_row];
                let row_weights = dequantize_q4_k_simd(row_data).expect("K dequant failed");
                let dot: f32 = row_weights
                    .iter()
                    .zip(normed.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                k_all_dots.push(dot);
            }
            let k_mean_abs_dot = k_all_dots.iter().map(|x| x.abs()).sum::<f32>() / k.out_dim as f32;
            let k_std_dot =
                (k_all_dots.iter().map(|x| x * x).sum::<f32>() / k.out_dim as f32).sqrt();

            // Compute all V dots
            let mut v_all_dots = Vec::with_capacity(v.out_dim);
            for row in 0..v.out_dim {
                let row_data = &v.data[row * v_bytes_per_row..(row + 1) * v_bytes_per_row];
                let row_weights = dequantize_q6_k(row_data).expect("V dequant failed");
                let dot: f32 = row_weights
                    .iter()
                    .zip(normed.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                v_all_dots.push(dot);
            }
            let v_mean_abs_dot = v_all_dots.iter().map(|x| x.abs()).sum::<f32>() / v.out_dim as f32;
            let v_std_dot =
                (v_all_dots.iter().map(|x| x * x).sum::<f32>() / v.out_dim as f32).sqrt();

            println!(
                "K: mean |dot| = {:.6}, std(dot) = {:.6}",
                k_mean_abs_dot, k_std_dot
            );
            println!(
                "V: mean |dot| = {:.6}, std(dot) = {:.6}",
                v_mean_abs_dot, v_std_dot
            );
            println!(
                "Ratio (K/V): mean |dot| = {:.1}x, std = {:.1}x",
                k_mean_abs_dot / v_mean_abs_dot,
                k_std_dot / v_std_dot
            );

            // Check if we can get sensible output by reinterpreting V as K (Q4_K)
            println!("\n=== What if V data was interpreted as Q4_K? ===");
            // V is Q6_K with 256 rows × 8 superblocks × 210 bytes = 430080 bytes
            // If it were Q4_K: 256 rows × 8 superblocks × 144 bytes = 294912 bytes
            // The first 294912 bytes of V data reinterpreted as Q4_K:
            let fake_k_bytes_per_row = (k.in_dim / 256) * 144; // Same as K
            println!("Treating V data as Q4_K:");
            for row in 0..5.min(v.out_dim) {
                let row_start = row * fake_k_bytes_per_row;
                if row_start + fake_k_bytes_per_row <= v.data.len() {
                    let row_data = &v.data[row_start..row_start + fake_k_bytes_per_row];
                    match dequantize_q4_k_simd(row_data) {
                        Ok(row_weights) => {
                            let dot: f32 = row_weights
                                .iter()
                                .zip(normed.iter())
                                .map(|(w, x)| w * x)
                                .sum();
                            let cos = dot / (l2_norm(&row_weights) * normed_l2);
                            println!(
                                "  V-as-Q4K row {:3}: dot={:+.6}, cos={:+.6}, L2={:.4}",
                                row,
                                dot,
                                cos,
                                l2_norm(&row_weights)
                            );
                        },
                        Err(e) => println!("  V-as-Q4K row {}: error: {:?}", row, e),
                    }
                }
            }
        },
        _ => println!("QKV is fused"),
    }

    println!("\n=== Analysis complete ===");
}
