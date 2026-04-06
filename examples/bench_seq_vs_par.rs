//! Benchmark sequential vs parallel matmul for Qwen2.5 dimensions
//! PAR-126: Determine if Rayon overhead makes sequential faster
use realizar::quantize::{
    fused_q4k_parallel_matvec_into, fused_q4k_q8k_parallel_matvec_into,
    quantize_activations_q8k_into,
};
use std::time::Instant;

// Sequential Q4K matmul (no Rayon)
fn q4k_sequential_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    let super_blocks = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks * 144;

    for row in 0..out_dim {
        let row_data = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
        let mut sum = 0.0f32;

        for sb in 0..super_blocks {
            let sb_data = &row_data[sb * 144..(sb + 1) * 144];

            // Parse Q4_K super-block
            let d = f16_to_f32(u16::from_le_bytes([sb_data[0], sb_data[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([sb_data[2], sb_data[3]]));
            let scales = &sb_data[4..16];
            let quants = &sb_data[16..144];

            // Process 8 blocks of 32 elements each
            for block in 0..8 {
                let scale_byte = scales[(block + (block / 2)) & !1];
                let sc = (scale_byte & 0x3F) as f32;
                let m = ((scale_byte >> 4)
                    | ((scales[block / 2 + 8] >> (4 * (block % 2))) << 4) & 0x3F)
                    as f32;

                let block_quants = &quants[block * 16..(block + 1) * 16];
                let act_offset = sb * 256 + block * 32;

                for i in 0..16 {
                    let q_byte = block_quants[i];
                    let q_lo = (q_byte & 0x0F) as f32;
                    let q_hi = (q_byte >> 4) as f32;

                    if act_offset + i < in_dim {
                        sum += activations[act_offset + i] * (d * sc * q_lo - dmin * m);
                    }
                    if act_offset + 16 + i < in_dim {
                        sum += activations[act_offset + 16 + i] * (d * sc * q_hi - dmin * m);
                    }
                }
            }
        }
        output[row] = sum;
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormal
        let mut m = mant;
        let mut e = 1u32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = 127 - 15 - e + 1;
        let mant32 = (m & 0x3FF) << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::NAN
        }
    } else {
        let exp32 = exp + 127 - 15;
        let mant32 = mant << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
    }
}

fn main() -> Result<(), realizar::RealizarError> {
    // Qwen2.5-Coder-1.5B dimensions
    let hidden_dim: usize = 1536;
    let intermediate_dim: usize = 8960;

    println!("=== Sequential vs Parallel Matmul Benchmark ===\n");

    // Test different matmul sizes in transformer
    let test_cases = [
        (
            "QKV (1536→1536+384+384)",
            hidden_dim,
            hidden_dim + 384 + 384,
        ),
        ("Attn Out (1536→1536)", hidden_dim, hidden_dim),
        ("FFN Up (1536→8960)", hidden_dim, intermediate_dim),
        ("FFN Down (8960→1536)", intermediate_dim, hidden_dim),
    ];

    for (name, in_dim, out_dim) in test_cases {
        println!("--- {} ---", name);

        // Create test data
        let super_blocks = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks * 144;
        let weight_bytes = out_dim * bytes_per_row;
        let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
        let activations: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 / in_dim as f32) * 2.0 - 1.0)
            .collect();
        let mut output = vec![0.0f32; out_dim];

        let iterations = 50;

        // Warmup
        let _ =
            fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);

        // Sequential
        let start = Instant::now();
        for _ in 0..iterations {
            q4k_sequential_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
        }
        let seq_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Parallel (current)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = fused_q4k_parallel_matvec_into(
                &weights,
                &activations,
                in_dim,
                out_dim,
                &mut output,
            );
        }
        let par_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Q8K parallel (with pre-quantized activations)
        let padded_len = in_dim.next_multiple_of(256);
        let num_sb = padded_len / 256;
        let mut q8k_scales = vec![0.0f32; num_sb];
        let mut q8k_quants = vec![0i8; padded_len];
        quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants)?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = fused_q4k_q8k_parallel_matvec_into(
                &weights,
                &q8k_scales,
                &q8k_quants,
                in_dim,
                out_dim,
                &mut output,
            );
        }
        let q8k_us = start.elapsed().as_micros() as f64 / iterations as f64;

        println!("  Sequential:   {:>8.0} us", seq_us);
        println!(
            "  Parallel:     {:>8.0} us ({:.1}x)",
            par_us,
            seq_us / par_us
        );
        println!(
            "  Q8K Parallel: {:>8.0} us ({:.1}x)",
            q8k_us,
            seq_us / q8k_us
        );

        // Which is faster?
        let best = seq_us.min(par_us).min(q8k_us);
        let best_name = if best == seq_us {
            "Sequential"
        } else if best == par_us {
            "Parallel"
        } else {
            "Q8K Parallel"
        };
        println!("  Winner: {} (saves {:.0} us)\n", best_name, seq_us - best);
    }

    // Estimate full forward pass
    println!("=== Full Forward Pass Estimate (28 layers) ===");
    let ops_per_layer = 5; // QKV + Attn Out + Up + Gate + Down
    let rayon_overhead_us = 133.0; // From profiling
    let total_rayon_overhead_ms = ops_per_layer as f64 * 28.0 * rayon_overhead_us / 1000.0;
    println!(
        "Rayon overhead: {:.1} ms ({} dispatches)",
        total_rayon_overhead_ms,
        ops_per_layer * 28
    );

    Ok(())
}
