//! Benchmark: AVX-512 VNNI vs AVX2 for Q4K×Q8K dot products
//! Measures per-call latency and identifies the bottleneck

use std::time::Instant;

fn main() {
    // 1.5B model: 1536 hidden dim = 6 super-blocks
    let super_blocks = 6;
    let bytes_per_sb = 144;
    let values_per_sb = 256;

    let weight_bytes = super_blocks * bytes_per_sb;
    let activation_len = super_blocks * values_per_sb;

    println!("Q4K×Q8K VNNI vs AVX2 Benchmark");
    println!("  Super-blocks: {}", super_blocks);
    println!("  Weight bytes: {}", weight_bytes);
    println!("  Activations: {}", activation_len);

    // Create realistic test data
    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..activation_len)
        .map(|i| (i as f32 / activation_len as f32) * 2.0 - 1.0)
        .collect();

    // Pre-quantize activations to Q8K (like we'd do in real inference)
    let (q8k_scales, q8k_quants) = quantize_to_q8k(&activations);

    println!(
        "\nQ8K quantized: {} scales, {} quants",
        q8k_scales.len(),
        q8k_quants.len()
    );

    // Warmup
    for _ in 0..1000 {
        let _ = realizar::quantize::fused_q4k_q8k_dot_simd(&weights, &q8k_scales, &q8k_quants);
    }

    // Benchmark high-level dispatch (should use VNNI)
    let iters = 100000;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = realizar::quantize::fused_q4k_q8k_dot_simd(&weights, &q8k_scales, &q8k_quants);
    }
    let elapsed = start.elapsed();
    let per_iter_ns = elapsed.as_nanos() as f64 / iters as f64;

    println!("\nfused_q4k_q8k_dot_simd (dispatched):");
    println!("  Per call: {:.1} ns", per_iter_ns);

    // Calculate throughput
    let macs = (activation_len * 2) as f64; // each value: mul + add
    let gmacs = macs * 1e9 / per_iter_ns / 1e9;
    println!("  Throughput: {:.2} GMAC/s", gmacs);

    // Estimate matmul (8960 rows)
    let rows = 8960;
    let matmul_us = per_iter_ns * rows as f64 / 1000.0;
    println!("\nProjected matmul ({} rows): {:.1} µs", rows, matmul_us);

    // But actual parallel matmul:
    let input: Vec<f32> = (0..1536).map(|i| (i as f32 / 1536.0) * 2.0 - 1.0).collect();
    let matmul_weights = vec![0u8; 8960 * 864]; // 864 = 6 * 144

    // Warmup parallel
    let mut out = vec![0.0f32; 8960];
    for _ in 0..3 {
        let _ = realizar::quantize::fused_q4k_parallel_matvec_into(
            &matmul_weights,
            &input,
            1536,
            8960,
            &mut out,
        );
    }

    let iters2 = 100;
    let start2 = Instant::now();
    for _ in 0..iters2 {
        let _ = realizar::quantize::fused_q4k_parallel_matvec_into(
            &matmul_weights,
            &input,
            1536,
            8960,
            &mut out,
        );
    }
    let elapsed2 = start2.elapsed();
    let actual_matmul_us = elapsed2.as_micros() as f64 / iters2 as f64;

    println!("\nActual parallel matmul: {:.1} µs", actual_matmul_us);
    println!("  Sequential estimate: {:.1} µs", matmul_us);
    println!("  Parallel speedup: {:.1}x", matmul_us / actual_matmul_us);

    // Target comparison
    let target_tok_s = 290.0;
    let matmuls_per_token = 28 * 7; // 196
    let target_matmul_us = 1e6 / target_tok_s / matmuls_per_token as f64;
    println!(
        "\nTarget: {:.1} µs per matmul ({:.0} tok/s)",
        target_matmul_us, target_tok_s
    );
    println!("Gap: {:.1}x slower", actual_matmul_us / target_matmul_us);
}

/// Quantize f32 activations to Q8_K format (per-super-block scale + i8 values)
fn quantize_to_q8k(values: &[f32]) -> (Vec<f32>, Vec<i8>) {
    const QK_K: usize = 256;
    let num_sb = values.len() / QK_K;

    let mut scales = Vec::with_capacity(num_sb);
    let mut quants = Vec::with_capacity(values.len());

    for sb in 0..num_sb {
        let start = sb * QK_K;
        let end = start + QK_K;
        let chunk = &values[start..end];

        // Find max absolute value for scale
        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inv_scale = if scale > 0.0 { 127.0 / amax } else { 0.0 };

        scales.push(scale);

        // Quantize values
        for v in chunk {
            let q = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            quants.push(q);
        }
    }

    (scales, quants)
}
