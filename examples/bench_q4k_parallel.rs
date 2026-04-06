//! Benchmark parallel Q4K matmul
use realizar::quantize::fused_q4k_parallel_matvec;
use std::time::Instant;

fn main() {
    // 1.5B model: hidden=1536, intermediate=8960
    let hidden: usize = 1536;
    let inter: usize = 8960;

    // Q4_K: 144 bytes per super-block, 256 elements per super-block
    let super_blocks_per_row = hidden.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;
    let weight_bytes = inter * bytes_per_row;

    println!("Q4K Parallel Matmul Benchmark");
    println!("  Hidden: {}, Intermediate: {}", hidden, inter);
    println!(
        "  Weight bytes: {} ({:.1} MB)",
        weight_bytes,
        weight_bytes as f64 / 1e6
    );

    // Create test data
    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();

    // Warmup
    for _ in 0..10 {
        let _ = fused_q4k_parallel_matvec(&weights, &activations, hidden, inter);
    }

    // Benchmark
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = fused_q4k_parallel_matvec(&weights, &activations, hidden, inter);
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / iters as f64;

    println!("\nResults (FFN up/gate: {}→{}):", hidden, inter);
    println!("  Per matmul: {:.1} µs", per_iter_us);

    // 7 matmuls per layer of varying sizes
    // Approximate: 3 large (FFN), 2 medium (QKV), 2 small (output, down)
    let matmuls_per_layer = 7;
    let layers = 28;

    // More accurate estimate using actual dimensions
    // QKV: hidden → hidden*3 = 1536 → 4608
    // Output: hidden → hidden = 1536 → 1536
    // FFN up/gate: hidden → inter = 1536 → 8960
    // FFN down: inter → hidden = 8960 → 1536

    let estimated_token_ms = per_iter_us * matmuls_per_layer as f64 * layers as f64 / 1000.0;
    println!(
        "  Estimated per token ({} layers × {} matmuls): {:.1} ms = {:.1} tok/s",
        layers,
        matmuls_per_layer,
        estimated_token_ms,
        1000.0 / estimated_token_ms
    );

    // Compare with Ollama
    let ollama_tok_s = 290.0;
    let target_token_ms = 1000.0 / ollama_tok_s;
    println!(
        "\nTarget: {:.1} ms/token ({:.0} tok/s)",
        target_token_ms, ollama_tok_s
    );
    println!("Gap: {:.1}x slower", estimated_token_ms / target_token_ms);

    // Also show per-element throughput
    let ops = hidden * inter; // multiply-adds
    let gflops = (ops as f64 * 2.0) / (per_iter_us * 1e3); // GFLOPS (2 ops per MAC)
    println!("\nThroughput: {:.1} GFLOPS", gflops);
}
