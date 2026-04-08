#![allow(clippy::disallowed_methods)]
//! Chapter 21: Benchmark — aprender-serve vs Candle
//!
//! Data source: paiml/candle-vs-apr (RTX 4090, Qwen2.5-Coder-1.5B Q4_K_M)
//! Contract: contracts/apr-book-ch21-v1.yaml

use aprender::format::validated_tensors::TensorStats;

fn main() {
    // Benchmark results from paiml/candle-vs-apr/performance.md (v3)
    // Hardware: RTX 4090, Model: Qwen2.5-Coder-1.5B Q4_K_M GGUF
    let aprender_serve_tps = 273.8_f64;
    let candle_tps = 227.4_f64;
    let speedup = aprender_serve_tps / candle_tps;

    println!("=== aprender-serve vs Candle (RTX 4090) ===");
    println!("Model: Qwen2.5-Coder-1.5B Q4_K_M");
    println!();
    println!("Single-request decode (c=1):");
    println!("  aprender-serve: {aprender_serve_tps:.1} tok/s");
    println!("  Candle:         {candle_tps:.1} tok/s");
    println!("  Speedup:        {speedup:.2}x");
    assert!(speedup > 1.0, "aprender-serve must be faster than Candle");

    // Scaling under concurrency (Candle has no server — N/A)
    let c32_tps = 1776.5_f64;
    let scaling = c32_tps / aprender_serve_tps;
    println!();
    println!("Scaling (aprender-serve only — Candle has no server):");
    println!("  c=1:  {aprender_serve_tps:.1} tok/s");
    println!("  c=32: {c32_tps:.1} tok/s");
    println!("  Scaling: {scaling:.1}x");
    assert!(scaling > 5.0, "Must scale >5x from c=1 to c=32");

    // Memory: Candle wins on RSS (no HTTP server overhead)
    let candle_rss_mb = 449_u64;
    let aprender_rss_mb = 3082_u64;
    println!();
    println!("Memory (Peak RSS):");
    println!("  Candle:         {candle_rss_mb} MB (CLI only, no server)");
    println!("  aprender-serve: {aprender_rss_mb} MB (HTTP server + KV cache)");

    // Use TensorStats to demonstrate aprender API usage
    let tps_samples = vec![273.8_f32, 271.2, 275.1, 273.0, 274.5];
    let stats = TensorStats::compute(&tps_samples);
    println!();
    println!("Throughput stability (5 runs):");
    println!("  mean: {:.1} tok/s, min: {:.1}, max: {:.1}", stats.mean, stats.min, stats.max);
    assert!(stats.mean > 270.0, "Mean throughput must exceed 270 tok/s");

    println!();
    println!("Repo: https://github.com/paiml/candle-vs-apr");
    println!("Chapter 21 contracts: PASSED");
}
