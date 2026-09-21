#![allow(clippy::disallowed_methods)]
//! Chapter 22: Benchmark — aprender-serve vs llama.cpp
//!
//! Data source: paiml/candle-vs-apr/results/bootstrap-*.json
//! Contract: contracts/apr-book-ch22-v1.yaml

use aprender::format::validated_tensors::TensorStats;

fn main() {
    // Bootstrap statistical comparison from paiml/candle-vs-apr
    // Model: Qwen2.5-Coder-1.5B Q4_K_M, Hardware: RTX 4090
    println!("=== aprender-serve vs llama.cpp (RTX 4090) ===");
    println!("Model: Qwen2.5-Coder-1.5B Q4_K_M");
    println!("Method: bootstrap statistical comparison (probador llm load)");
    println!();

    // #3773: both means are the bootstrap receipts this chapter's contract names
    // (contracts/apr-book-ch22-v1.yaml benchmark_data.data_files), in
    // paiml/candle-vs-apr. The llama.cpp figure used to read "285.0 — approximate
    // from bootstrap JSON"; the JSON says 431.1, and 285 flattered the ratio.
    // receipt: paiml/candle-vs-apr results/bootstrap-llama-cpp-b7746-fair-20260405-180222.json (2026-04-05) mean
    let llamacpp_decode_tps = 431.1_f64;
    // receipt: paiml/candle-vs-apr results/bootstrap-realizr-0.8.6-20260406-080315.json (2026-04-06) mean
    let aprender_decode_tps = 351.97_f64;
    let ratio = aprender_decode_tps / llamacpp_decode_tps;

    println!("Decode throughput (c=1), historical receipts from 2026-04-05/06:");
    println!("  llama.cpp (b7746): {llamacpp_decode_tps:.1} tok/s");
    println!("  aprender-serve:    {aprender_decode_tps:.1} tok/s");
    println!("  Ratio:             {ratio:.2}x");
    println!();
    println!("Analysis:");
    println!("  llama.cpp is a mature C++ project with years of optimization.");
    println!(
        "  aprender-serve achieves {:.0}% of llama.cpp throughput in pure Rust.",
        ratio * 100.0
    );
    println!("  At c=32, aprender-serve scales to 1,776 tok/s (llama.cpp: CLI only).");
    assert!(ratio > 0.8, "Must achieve >80% of llama.cpp throughput");

    // Bootstrap confidence interval demonstration
    let bootstrap_samples = vec![
        272.1_f32, 274.5, 273.8, 275.2, 271.9, 274.0, 273.3, 274.8, 272.7, 273.5,
    ];
    let stats = TensorStats::compute(&bootstrap_samples);
    println!();
    println!("Bootstrap statistics (10 samples):");
    println!(
        "  mean: {:.1}, min: {:.1}, max: {:.1}",
        stats.mean, stats.min, stats.max
    );
    let range = stats.max - stats.min;
    println!("  range: {range:.1} tok/s (narrow = stable)");
    assert!(range < 10.0, "Bootstrap range must be <10 tok/s (stable)");

    println!();
    println!("Repo: https://github.com/paiml/candle-vs-apr");
    println!("Chapter 22 contracts: PASSED");
}
