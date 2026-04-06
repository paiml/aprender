//! Q4K×Q8K and Q6K×f32 micro-benchmark comparison
use realizar::quantize::{
    fused_q4k_q8k_parallel_matvec_into, fused_q6k_parallel_matvec_into,
    quantize_activations_q8k_into, QK_K,
};
use std::time::Instant;

fn main() {
    // Qwen2.5-1.5B FFN down dimensions
    let in_dim: usize = 8960; // intermediate_dim
    let out_dim: usize = 1536; // hidden_dim

    const Q4K_SUPER_BLOCK_BYTES: usize = 144;
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row_q4k = super_blocks_per_row * Q4K_SUPER_BLOCK_BYTES;

    const Q6K_SUPER_BLOCK_BYTES: usize = 210;
    let bytes_per_row_q6k = super_blocks_per_row * Q6K_SUPER_BLOCK_BYTES;

    // Create random Q4K weights
    let weight_data_q4k: Vec<u8> = (0..out_dim * bytes_per_row_q4k)
        .map(|i| (i % 256) as u8)
        .collect();

    // Create random Q6K weights
    let weight_data_q6k: Vec<u8> = (0..out_dim * bytes_per_row_q6k)
        .map(|i| (i % 256) as u8)
        .collect();

    // Create random activations
    let activations: Vec<f32> = (0..in_dim)
        .map(|i| (i as f32 * 0.001).sin() * 2.0)
        .collect();

    // Quantize to Q8K
    let padded_len = in_dim.next_multiple_of(256);
    let num_superblocks = padded_len / 256;
    let mut q8k_scales = vec![0.0f32; num_superblocks];
    let mut q8k_quants = vec![0i8; padded_len];
    quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants).unwrap();

    let mut output = vec![0.0f32; out_dim];

    // Warmup Q4K×Q8K
    for _ in 0..3 {
        fused_q4k_q8k_parallel_matvec_into(
            &weight_data_q4k,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        )
        .unwrap();
    }

    // Benchmark Q4K×Q8K
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        fused_q4k_q8k_parallel_matvec_into(
            &weight_data_q4k,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        )
        .unwrap();
    }
    let q8k_time = start.elapsed();
    let q8k_avg = q8k_time.as_micros() as f64 / iters as f64;

    // Benchmark Q6K×f32 (old path)
    let start = Instant::now();
    for _ in 0..iters {
        fused_q6k_parallel_matvec_into(
            &weight_data_q6k,
            &activations,
            in_dim,
            out_dim,
            &mut output,
        )
        .unwrap();
    }
    let f32_time = start.elapsed();
    let f32_avg = f32_time.as_micros() as f64 / iters as f64;

    println!("Quantized Matmul Benchmark Results");
    println!("==================================");
    println!("Matrix: {}x{}", out_dim, in_dim);
    println!();
    println!("Q6K×f32:        {:>8.1} µs/matmul", f32_avg);
    println!("Q4K×Q8K:        {:>8.1} µs/matmul", q8k_avg);
    println!();
    println!("Speedup: {:.2}x", f32_avg / q8k_avg);

    // GFLOPS
    let flops = 2.0 * out_dim as f64 * in_dim as f64;
    let f32_gflops = flops / (f32_avg * 1000.0);
    let q8k_gflops = flops / (q8k_avg * 1000.0);
    println!();
    println!("Throughput:");
    println!("  Q6K×f32:  {:>6.2} GFLOPS", f32_gflops);
    println!("  Q4K×Q8K:  {:>6.2} GFLOPS", q8k_gflops);
}
