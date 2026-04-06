//! Microbenchmark for Q4K SIMD dot product
use realizar::quantize::fused_q4k_dot_simd;
use std::time::Instant;

fn main() {
    // 1.5B model hidden_dim=1536, 6 super-blocks per row
    let super_blocks = 6;
    let bytes_per_sb = 144;
    let values_per_sb = 256;

    let weight_bytes = super_blocks * bytes_per_sb;
    let activation_len = super_blocks * values_per_sb;

    println!("Q4K SIMD Benchmark");
    println!("  Weight bytes: {}", weight_bytes);
    println!("  Activation len: {}", activation_len);

    // Create test data
    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..activation_len)
        .map(|i| (i as f32 / activation_len as f32) * 2.0 - 1.0)
        .collect();

    // Warmup
    for _ in 0..1000 {
        let _ = fused_q4k_dot_simd(&weights, &activations);
    }

    // Benchmark
    let iters = 10000;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = fused_q4k_dot_simd(&weights, &activations);
    }
    let elapsed = start.elapsed();
    let per_iter_ns = elapsed.as_nanos() as f64 / iters as f64;

    println!(
        "\nResults (single dot product, {} elements):",
        activation_len
    );
    println!("  Per call: {:.1} ns", per_iter_ns);

    // For full 1.5B matmul: 8960 rows (FFN up/gate)
    let ffn_rows = 8960;
    let estimated_matmul_us = per_iter_ns * ffn_rows as f64 / 1000.0;
    println!(
        "  Estimated FFN matmul ({} rows): {:.1} µs",
        ffn_rows, estimated_matmul_us
    );

    // 28 layers × 7 matmuls per layer
    let matmuls_per_token = 28 * 7;
    let estimated_token_ms = estimated_matmul_us * matmuls_per_token as f64 / 1000.0;
    println!(
        "  Estimated per token ({} matmuls): {:.1} ms = {:.1} tok/s",
        matmuls_per_token,
        estimated_token_ms,
        1000.0 / estimated_token_ms
    );

    // Compare with target
    let ollama_tok_s = 290.0;
    let target_token_ms = 1000.0 / ollama_tok_s;
    println!(
        "\nTarget: {:.1} ms/token ({:.0} tok/s)",
        target_token_ms, ollama_tok_s
    );
    println!("Gap: {:.1}x slower", estimated_token_ms / target_token_ms);
}
