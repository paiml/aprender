//! Benchmark V2 AVX-512 kernel directly

use realizar::quantize::{fused_q4k_q8k_dot_simd, quantize_activations_q8k_into};
use std::time::Instant;

fn main() {
    // Simulate a Q4K weight row (6 super-blocks for in_dim=1536)
    let in_dim: usize = 1536;
    let super_blocks = in_dim.div_ceil(256);
    let weight_data_size = super_blocks * 144;
    let weight_data: Vec<u8> = (0..weight_data_size).map(|i| (i % 256) as u8).collect();

    // Create f32 activations
    let activations: Vec<f32> = (0..in_dim)
        .map(|i| (i as f32 / in_dim as f32) * 2.0 - 1.0)
        .collect();

    // Quantize to Q8K
    let padded_len = in_dim.next_multiple_of(256);
    let num_superblocks = padded_len / 256;
    let mut q8k_scales = vec![0.0f32; num_superblocks];
    let mut q8k_quants = vec![0i8; padded_len];
    quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants).unwrap();

    // Warmup
    for _ in 0..1000 {
        let _ = fused_q4k_q8k_dot_simd(&weight_data, &q8k_scales, &q8k_quants);
    }

    // Benchmark
    let iterations = 50000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_q8k_dot_simd(&weight_data, &q8k_scales, &q8k_quants);
    }
    let elapsed = start.elapsed();

    let per_call_us = elapsed.as_micros() as f64 / iterations as f64;
    println!("=== Q4K×Q8K Dot Product (V2 Kernel) ===");
    println!("in_dim: {}", in_dim);
    println!("Super-blocks: {}", super_blocks);
    println!("Time per call: {:.3} µs", per_call_us);

    // Estimate full matmul (1536 output rows)
    let out_dim: usize = 1536;
    let matmul_time_us = per_call_us * out_dim as f64;
    let matmul_time_ms = matmul_time_us / 1000.0;
    println!(
        "\nEstimated matmul time ({}x{}): {:.2} µs = {:.3} ms",
        out_dim, in_dim, matmul_time_us, matmul_time_ms
    );

    // Compare with expected
    println!("\n=== Comparison ===");
    println!("Before V2 (horizontal sums in loop): ~225 µs per matmul");
    println!(
        "V2 kernel (deferred horizontal sums): {:.1} µs per matmul",
        matmul_time_ms * 1000.0
    );
}
