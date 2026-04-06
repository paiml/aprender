//! Detailed matmul profiling to identify optimization opportunities
//!
//! Run with: cargo run --release --example profile_matmul_detail

use realizar::quantize::{fused_q4_0_q8_0_parallel_matvec, quantize_activations_q8_0};
use std::time::Instant;

fn main() {
    println!("=== Detailed Matmul Profiling ===\n");

    // Test dimensions from real models
    let test_cases = [
        ("Qwen2.5 QKV", 896, 1152),       // small
        ("Qwen2.5 FFN up", 896, 4864),    // medium
        ("Qwen2.5 LM head", 896, 151936), // large (bottleneck!)
        ("TinyLlama QKV", 2048, 2560),
        ("TinyLlama FFN up", 2048, 5632),
        ("TinyLlama LM head", 2048, 32000),
    ];

    for (name, in_dim, out_dim) in test_cases {
        profile_matmul_detailed(name, in_dim, out_dim);
    }

    // Test threading overhead
    println!("\n=== Threading Overhead Analysis ===\n");
    test_threading_overhead();

    // Test cache effects
    println!("\n=== Cache Effect Analysis ===\n");
    test_cache_effects();
}

fn profile_matmul_detailed(name: &str, in_dim: usize, out_dim: usize) {
    const Q4_0_BLOCK_SIZE: usize = 32;
    const Q4_0_BLOCK_BYTES: usize = 18;

    let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
    let weight_bytes = out_dim * bytes_per_row;

    // Create test data
    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Warmup
    for _ in 0..3 {
        let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
    }

    // Profile: Total time
    let mut total_times = Vec::new();
    for _ in 0..20 {
        let start = Instant::now();
        let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
        total_times.push(start.elapsed());
    }

    // Profile: Just quantization
    let mut quant_times = Vec::new();
    for _ in 0..20 {
        let start = Instant::now();
        let _ = quantize_activations_q8_0(&activations);
        quant_times.push(start.elapsed());
    }

    let total_us: Vec<u128> = total_times.iter().map(|t| t.as_micros()).collect();
    let quant_us: Vec<u128> = quant_times.iter().map(|t| t.as_micros()).collect();

    let total_median = median(&total_us);
    let quant_median = median(&quant_us);
    let compute_median = total_median.saturating_sub(quant_median);

    let weight_mb = weight_bytes as f64 / 1e6;
    let bandwidth_gbps = weight_mb / 1e3 / (total_median as f64 / 1e6);

    // FLOP analysis (2 ops per multiply-add)
    let flops = 2.0 * in_dim as f64 * out_dim as f64;
    let gflops = flops / 1e9 / (total_median as f64 / 1e6);

    println!("{} ({}×{}):", name, in_dim, out_dim);
    println!("  Weights:     {:>8.2} MB", weight_mb);
    println!("  Total:       {:>8} µs", total_median);
    println!(
        "    Quantize:  {:>8} µs ({:.1}%)",
        quant_median,
        100.0 * quant_median as f64 / total_median as f64
    );
    println!(
        "    Compute:   {:>8} µs ({:.1}%)",
        compute_median,
        100.0 * compute_median as f64 / total_median as f64
    );
    println!("  Bandwidth:   {:>8.1} GB/s", bandwidth_gbps);
    println!("  GFLOP/s:     {:>8.1}", gflops);
    println!();
}

fn test_threading_overhead() {
    // Compare single-threaded vs multi-threaded for small vs large workloads
    let in_dim: usize = 896;

    for out_dim in [64, 256, 1024, 4096, 16384] {
        const Q4_0_BLOCK_SIZE: usize = 32;
        const Q4_0_BLOCK_BYTES: usize = 18;
        let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
        let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
        let weight_bytes = out_dim * bytes_per_row;

        let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
        let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

        // Warmup
        for _ in 0..3 {
            let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
        }

        let mut times = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
            times.push(start.elapsed().as_micros());
        }

        let med = median(&times);
        let throughput = out_dim as f64 / (med as f64 / 1e6); // rows/sec
        println!(
            "  out_dim={:>6}: {:>6} µs ({:.0} rows/s)",
            out_dim, med, throughput
        );
    }
}

fn test_cache_effects() {
    // Test how cache size affects performance
    let in_dim: usize = 2048;
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // L1 cache: 32KB, L2: 256KB, L3: varies
    // Q4_0 row size for 2048 input = 64 blocks * 18 bytes = 1152 bytes

    for &weight_kb in &[16, 64, 256, 1024, 4096, 16384] {
        const Q4_0_BLOCK_BYTES: usize = 18;
        const Q4_0_BLOCK_SIZE: usize = 32;
        let blocks_per_row = in_dim.div_ceil(Q4_0_BLOCK_SIZE);
        let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
        let out_dim = (weight_kb * 1024) / bytes_per_row;
        let weight_bytes = out_dim * bytes_per_row;

        let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();

        // Warmup
        for _ in 0..3 {
            let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
        }

        let mut times = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
            times.push(start.elapsed().as_micros());
        }

        let med = median(&times);
        let weight_mb = weight_bytes as f64 / 1e6;
        let bandwidth_gbps = weight_mb / 1e3 / (med as f64 / 1e6);
        println!(
            "  {:>5} KB weights: {:>6} µs, {:.1} GB/s",
            weight_kb, med, bandwidth_gbps
        );
    }
}

fn median(times: &[u128]) -> u128 {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}
