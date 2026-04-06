//! PAR-001c: Check V weight dequantization
//!
//! The V output is ~50x smaller than Q and K. Let's verify the V weights are correct.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q6_k;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001c: V Weight Analysis ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[0];

    // Get QKV weights info
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!(
                "Q weight: qtype={}, in_dim={}, out_dim={}, data_len={}",
                q.qtype,
                q.in_dim,
                q.out_dim,
                q.data.len()
            );
            println!(
                "K weight: qtype={}, in_dim={}, out_dim={}, data_len={}",
                k.qtype,
                k.in_dim,
                k.out_dim,
                k.data.len()
            );
            println!(
                "V weight: qtype={}, in_dim={}, out_dim={}, data_len={}",
                v.qtype,
                v.in_dim,
                v.out_dim,
                v.data.len()
            );

            // Try to dequantize V (Q6_K)
            println!("\nDequantizing V weight (Q6_K)...");
            match dequantize_q6_k(&v.data) {
                Ok(v_dequant) => {
                    let expected_elements = v.in_dim * v.out_dim;
                    println!(
                        "  Dequantized elements: {} (expected: {})",
                        v_dequant.len(),
                        expected_elements
                    );
                    println!("  L2 norm: {:.4}", l2_norm(&v_dequant));
                    println!("  First 10: {:?}", &v_dequant[..10.min(v_dequant.len())]);
                    println!(
                        "  Last 10: {:?}",
                        &v_dequant[v_dequant.len().saturating_sub(10)..]
                    );

                    // Check first row (should be meaningful weights)
                    let first_row = &v_dequant[..v.in_dim.min(v_dequant.len())];
                    println!("\n  First row stats:");
                    println!("    L2: {:.4}", l2_norm(first_row));
                    let min = first_row.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = first_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    println!("    Range: [{:.6}, {:.6}]", min, max);
                },
                Err(e) => {
                    println!("  Failed to dequantize V: {:?}", e);
                },
            }

            // Also check Q and K for comparison
            println!("\nDequantizing Q weight (Q4_K) for comparison...");
            use realizar::quantize::dequantize_q4_k_simd;
            match dequantize_q4_k_simd(&q.data) {
                Ok(q_dequant) => {
                    println!("  Dequantized elements: {}", q_dequant.len());
                    println!("  L2 norm: {:.4}", l2_norm(&q_dequant));
                    let first_row = &q_dequant[..q.in_dim.min(q_dequant.len())];
                    println!("  First row L2: {:.4}", l2_norm(first_row));
                },
                Err(e) => {
                    println!("  Failed to dequantize Q: {:?}", e);
                },
            }
        },
        _ => {
            println!("QKV is fused, not separate");
        },
    }

    println!("\n=== Analysis complete ===");
}
