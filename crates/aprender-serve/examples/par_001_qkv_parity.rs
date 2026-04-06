//! PAR-001d: QKV parity test - compare byte-level weight interpretation
//!
//! The hypothesis: GGML stores weights in a different order than we're reading.
//! This test will:
//! 1. Extract raw Q, K, V weight bytes
//! 2. Dequantize manually with different interpretations
//! 3. Compare against llama.cpp expected values

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k_simd, dequantize_q6_k};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001d: QKV Weight Byte-Level Parity Test ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    println!("Config:");
    println!(
        "  hidden_dim={}, num_heads={}, num_kv_heads={}",
        hidden_dim, num_heads, num_kv_heads
    );
    println!("  head_dim={}, kv_dim={}", head_dim, kv_dim);

    // Get layer 0 weights
    let layer = &model.layers()[0];

    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k: _, v } => {
            println!("\nQ weight:");
            println!(
                "  in_dim={}, out_dim={}, qtype={}",
                q.in_dim, q.out_dim, q.qtype
            );
            println!("  data_len={} bytes", q.data.len());

            // Q4_K: 144 bytes per superblock (256 elements)
            let q_superblocks_per_row = q.in_dim.div_ceil(256);
            let q_bytes_per_row = q_superblocks_per_row * 144;
            let q_expected_bytes = q.out_dim * q_bytes_per_row;
            println!(
                "  Expected layout: {} rows × {} bytes/row = {} bytes",
                q.out_dim, q_bytes_per_row, q_expected_bytes
            );

            // Dequantize first row (row 0 of Q weight matrix)
            if q.data.len() >= q_bytes_per_row {
                let row0_data = &q.data[0..q_bytes_per_row];
                match dequantize_q4_k_simd(row0_data) {
                    Ok(row0_dequant) => {
                        println!(
                            "  Row 0 dequantized: {} elements, L2={:.4}",
                            row0_dequant.len(),
                            l2_norm(&row0_dequant)
                        );
                        println!(
                            "  Row 0 first 10: {:?}",
                            &row0_dequant[..10.min(row0_dequant.len())]
                        );
                    },
                    Err(e) => println!("  Failed to dequantize row 0: {:?}", e),
                }

                // Also try row 1
                if q.data.len() >= 2 * q_bytes_per_row {
                    let row1_data = &q.data[q_bytes_per_row..2 * q_bytes_per_row];
                    if let Ok(row1_dequant) = dequantize_q4_k_simd(row1_data) {
                        println!(
                            "  Row 1 L2={:.4}, first 5: {:?}",
                            l2_norm(&row1_dequant),
                            &row1_dequant[..5.min(row1_dequant.len())]
                        );
                    }
                }
            }

            println!("\nV weight:");
            println!(
                "  in_dim={}, out_dim={}, qtype={}",
                v.in_dim, v.out_dim, v.qtype
            );
            println!("  data_len={} bytes", v.data.len());

            // Q6_K: 210 bytes per superblock (256 elements)
            let v_superblocks_per_row = v.in_dim.div_ceil(256);
            let v_bytes_per_row = v_superblocks_per_row * 210;
            let v_expected_bytes = v.out_dim * v_bytes_per_row;
            println!(
                "  Expected layout: {} rows × {} bytes/row = {} bytes",
                v.out_dim, v_bytes_per_row, v_expected_bytes
            );

            // Dequantize first row
            if v.data.len() >= v_bytes_per_row {
                let row0_data = &v.data[0..v_bytes_per_row];
                match dequantize_q6_k(row0_data) {
                    Ok(row0_dequant) => {
                        println!(
                            "  Row 0 dequantized: {} elements, L2={:.4}",
                            row0_dequant.len(),
                            l2_norm(&row0_dequant)
                        );
                        println!(
                            "  Row 0 first 10: {:?}",
                            &row0_dequant[..10.min(row0_dequant.len())]
                        );
                    },
                    Err(e) => println!("  Failed to dequantize row 0: {:?}", e),
                }
            }

            // KEY TEST: Try interpreting V data with TRANSPOSED layout
            println!("\n=== Testing TRANSPOSED interpretation ===");
            println!("If GGML stores column-major, then:");
            println!(
                "  - We're reading {} 'rows' of {} elements each",
                v.out_dim, v.in_dim
            );
            println!(
                "  - But GGML might store {} 'columns' of {} elements each",
                v.in_dim, v.out_dim
            );

            // For column-major [in_dim, out_dim]:
            // Column 0 contains elements [0,0], [1,0], [2,0], ..., [in_dim-1, 0]
            // These would be used to compute output[0] from all input elements

            // In column-major, superblocks would be packed along columns (out_dim elements)
            // So each superblock contains 256 consecutive column elements
            let v_superblocks_per_col = v.out_dim.div_ceil(256); // 256/256 = 1
            let v_bytes_per_col = v_superblocks_per_col * 210; // 1 * 210 = 210
            let v_expected_bytes_transposed = v.in_dim * v_bytes_per_col;
            println!(
                "  Transposed layout: {} cols × {} bytes/col = {} bytes",
                v.in_dim, v_bytes_per_col, v_expected_bytes_transposed
            );

            // Try reading column 0 (first 210 bytes)
            if v.data.len() >= v_bytes_per_col {
                let col0_data = &v.data[0..v_bytes_per_col];
                match dequantize_q6_k(col0_data) {
                    Ok(col0_dequant) => {
                        println!(
                            "  'Column 0' dequantized: {} elements, L2={:.4}",
                            col0_dequant.len(),
                            l2_norm(&col0_dequant)
                        );
                        println!(
                            "  'Column 0' first 10: {:?}",
                            &col0_dequant[..10.min(col0_dequant.len())]
                        );
                    },
                    Err(e) => println!("  Failed to dequantize 'column 0': {:?}", e),
                }
            }

            // Print raw bytes for comparison
            println!("\n=== Raw byte comparison ===");
            println!(
                "V weight first 32 bytes: {:02x?}",
                &v.data[..32.min(v.data.len())]
            );
            println!(
                "Q weight first 32 bytes: {:02x?}",
                &q.data[..32.min(q.data.len())]
            );

            // Check d (scale) values at the start of each superblock
            println!("\n=== Superblock scale (d) analysis ===");
            // Q4_K superblock: d (2 bytes f16) + dmin (2 bytes f16) + scales (12 bytes) + qs (128 bytes)
            for sb in 0..4.min(q.out_dim) {
                let sb_start = sb * q_bytes_per_row; // Start of row sb
                if sb_start + 4 <= q.data.len() {
                    let d_bytes = [q.data[sb_start], q.data[sb_start + 1]];
                    let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
                    let dmin_bytes = [q.data[sb_start + 2], q.data[sb_start + 3]];
                    let dmin = half::f16::from_bits(u16::from_le_bytes(dmin_bytes)).to_f32();
                    println!("Q row {} superblock 0: d={:.6}, dmin={:.6}", sb, d, dmin);
                }
            }

            for sb in 0..4.min(v.out_dim) {
                let sb_start = sb * v_bytes_per_row; // Start of row sb
                                                     // Q6_K superblock: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (2 bytes)
                                                     // d is at offset 128 + 64 + 16 = 208
                if sb_start + 210 <= v.data.len() {
                    let d_offset = sb_start + 208;
                    let d_bytes = [v.data[d_offset], v.data[d_offset + 1]];
                    let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
                    println!("V row {} superblock 0: d={:.6}", sb, d);
                }
            }
        },
        _ => {
            println!("QKV is fused, not separate");
        },
    }

    // Test: Full QKV projection and analyze output structure
    println!("\n=== Full QKV Projection Analysis ===");
    let token_id: u32 = 26222; // "Once"
    let hidden = model.embed(&[token_id]);
    println!("Embedding L2: {:.4}", l2_norm(&hidden));

    // Apply RMS norm
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + model.config().eps).sqrt();
    let normed: Vec<f32> = hidden
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| x / rms * w)
        .collect();
    println!("Normed L2: {:.4}", l2_norm(&normed));

    // Do QKV projection
    let qkv = model
        .qkv_matmul(&normed, &layer.qkv_weight)
        .expect("QKV matmul failed");

    let q = &qkv[..hidden_dim];
    let k = &qkv[hidden_dim..hidden_dim + kv_dim];
    let v_out = &qkv[hidden_dim + kv_dim..];

    println!("\nQKV projection output:");
    println!(
        "  Q: len={}, L2={:.4}, first 5: {:?}",
        q.len(),
        l2_norm(q),
        &q[..5]
    );
    println!(
        "  K: len={}, L2={:.4}, first 5: {:?}",
        k.len(),
        l2_norm(k),
        &k[..5]
    );
    println!(
        "  V: len={}, L2={:.4}, first 5: {:?}",
        v_out.len(),
        l2_norm(v_out),
        &v_out[..5]
    );

    // Check for patterns - are all V outputs near zero?
    let v_nonzero = v_out.iter().filter(|&&x| x.abs() > 0.01).count();
    let k_nonzero = k.iter().filter(|&&x| x.abs() > 0.01).count();
    let q_nonzero = q.iter().filter(|&&x| x.abs() > 0.01).count();
    println!("\nNon-zero counts (|x| > 0.01):");
    println!(
        "  Q: {}/{} ({:.1}%)",
        q_nonzero,
        q.len(),
        100.0 * q_nonzero as f32 / q.len() as f32
    );
    println!(
        "  K: {}/{} ({:.1}%)",
        k_nonzero,
        k.len(),
        100.0 * k_nonzero as f32 / k.len() as f32
    );
    println!(
        "  V: {}/{} ({:.1}%)",
        v_nonzero,
        v_out.len(),
        100.0 * v_nonzero as f32 / v_out.len() as f32
    );

    println!("\n=== Analysis complete ===");
}
