//! Profile LM head performance
//!
//! Run with: cargo run --release --example profile_lm_head

use realizar::quantize::fused_q4_0_q8_0_parallel_matvec;
use std::time::Instant;

fn main() {
    println!("=== LM Head Performance Profile ===\n");

    // TinyLlama dimensions
    profile_matmul("TinyLlama LM head", 2048, 32_000);

    // Qwen2.5 dimensions
    profile_matmul("Qwen2.5 LM head", 896, 151_936);

    // Compare smaller matmuls (per-layer)
    println!("\n=== Per-Layer Matmul Comparison ===\n");

    // TinyLlama QKV
    profile_matmul("TinyLlama QKV", 2048, 2560);

    // Qwen2.5 QKV
    profile_matmul("Qwen2.5 QKV", 896, 1152);

    // TinyLlama FFN up
    profile_matmul("TinyLlama FFN up", 2048, 5632);

    // Qwen2.5 FFN up
    profile_matmul("Qwen2.5 FFN up", 896, 4864);
}

fn profile_matmul(name: &str, in_dim: usize, out_dim: usize) {
    // Create fake Q4_0 weights (18 bytes per block of 32 elements)
    let blocks_per_row = in_dim.div_ceil(32);
    let bytes_per_row = blocks_per_row * 18;
    let weight_bytes = out_dim * bytes_per_row;

    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Warmup
    for _ in 0..5 {
        let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
    }

    // Profile
    let mut times = Vec::new();
    for _ in 0..50 {
        let start = Instant::now();
        let _ = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, in_dim, out_dim);
        times.push(start.elapsed());
    }

    let times_us: Vec<u128> = times.iter().map(|t| t.as_micros()).collect();
    let min = *times_us.iter().min().expect("test");
    let max = *times_us.iter().max().expect("test");
    let sum: u128 = times_us.iter().sum();
    let _avg = sum / times_us.len() as u128;

    let mut sorted = times_us;
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];

    let flops = 2.0 * in_dim as f64 * out_dim as f64;
    let gflops_per_sec = flops / 1e9 / (median as f64 / 1e6);
    let weight_mb = weight_bytes as f64 / 1e6;
    let bandwidth_gbps = weight_mb / 1e3 / (median as f64 / 1e6);

    println!("{} ({}x{}):", name, in_dim, out_dim);
    println!("  Weights:   {:.1} MB", weight_mb);
    println!(
        "  Median:    {} µs ({:.2} ms)",
        median,
        median as f64 / 1000.0
    );
    println!("  Min/Max:   {}/{} µs", min, max);
    println!("  GFLOP/s:   {:.1}", gflops_per_sec);
    println!("  Bandwidth: {:.1} GB/s", bandwidth_gbps);
    println!();
}
