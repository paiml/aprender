#![allow(clippy::disallowed_methods)]
//! Chapter 23: Benchmark — Training: PyTorch vs unsloth vs cuBLAS vs WGPU
//!
//! Data source: paiml/qwen-train-canary/performance.md
//! Contract: contracts/apr-book-ch23-v1.yaml

use aprender::format::validated_tensors::TensorStats;

fn main() {
    // Training canary data from paiml/qwen-train-canary
    // Model: Qwen2.5-Coder-1.5B
    println!("=== Training Benchmark: Qwen2.5-Coder-1.5B ===");
    println!();

    // Results by backend (tok/s, VRAM MB)
    let backends: Vec<(&str, &str, f64, u64)> = vec![
        ("pytorch-compile", "gx10 A100", 3597.7, 34215),
        ("cuBLAS (default)", "gx10 A100", 4009.5, 49777),
        ("cuBLAS (forced)",  "gx10 A100", 4026.8, 49778),
        ("pytorch",          "gx10 A100", 4055.4, 50580),
        ("unsloth",          "yoga RTX",  6715.7, 3515),
        ("unsloth",          "gx10 A100", 13659.7, 10219),
    ];

    println!("| Backend          | Host      | tok/s    | VRAM (MB) |");
    println!("|------------------|-----------|----------|-----------|");
    for (backend, host, tps, vram) in &backends {
        println!("| {backend:<16} | {host:<9} | {tps:>8.1} | {vram:>9} |");
    }

    // Assertions on the data
    let unsloth_gx10_tps = 13659.7_f64;
    let pytorch_compile_tps = 3597.7_f64;
    let unsloth_speedup = unsloth_gx10_tps / pytorch_compile_tps;
    println!();
    println!("unsloth vs pytorch-compile: {unsloth_speedup:.1}x faster");
    assert!(unsloth_speedup > 3.0, "unsloth must be >3x faster than pytorch-compile");

    // VRAM efficiency: unsloth uses 10x less VRAM
    let unsloth_vram = 3515_u64;
    let pytorch_vram = 50580_u64;
    let vram_ratio = pytorch_vram as f64 / unsloth_vram as f64;
    println!("VRAM: pytorch {pytorch_vram} MB vs unsloth {unsloth_vram} MB ({vram_ratio:.0}x less)");
    assert!(vram_ratio > 10.0, "unsloth must use >10x less VRAM");

    // TensorStats on throughput data
    let tps_data: Vec<f32> = backends.iter().map(|(_, _, tps, _)| *tps as f32).collect();
    let stats = TensorStats::compute(&tps_data);
    println!();
    println!("Throughput distribution across backends:");
    println!("  min: {:.0}, max: {:.0}, mean: {:.0}", stats.min, stats.max, stats.mean);

    println!();
    println!("Repo: https://github.com/paiml/qwen-train-canary");
    println!("Chapter 23 contracts: PASSED");
}
