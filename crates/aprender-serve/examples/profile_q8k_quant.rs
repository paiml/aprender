//! PAR-126: Profile Q8K quantization overhead
//! The Q8K pre-quantization runs twice per layer for VNNI acceleration

use realizar::quantize::quantize_activations_q8k_into;
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let hidden_dim = 1536usize;
    let num_layers = 28;
    let iterations = 1000;

    println!("=== Q8K Quantization Profiler ===\n");
    println!("hidden_dim={}, layers={}", hidden_dim, num_layers);

    // Allocate buffers
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0)
        .collect();
    let hidden_sb = hidden_dim.next_multiple_of(256) / 256;
    let mut scales = vec![0.0f32; hidden_sb];
    let mut quants = vec![0i8; hidden_dim.next_multiple_of(256)];

    // Warmup
    for _ in 0..100 {
        quantize_activations_q8k_into(&input, &mut scales, &mut quants)?;
    }

    // Measure single call
    let start = Instant::now();
    for _ in 0..iterations {
        quantize_activations_q8k_into(&input, &mut scales, &mut quants)?;
    }
    let q8k_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Q8K quantization: {:.1} µs", q8k_us);

    // Per-token overhead
    let calls_per_token = num_layers * 2; // 2x per layer (QKV + FFN)
    let overhead_per_token_us = q8k_us * calls_per_token as f64;
    let overhead_per_token_ms = overhead_per_token_us / 1000.0;
    println!(
        "\nCalls per token: {} ({} layers × 2)",
        calls_per_token, num_layers
    );
    println!(
        "Total Q8K overhead: {:.1} µs = {:.2} ms per token",
        overhead_per_token_us, overhead_per_token_ms
    );

    // Impact on gap
    let actual_ms = 31.0; // Current performance
    let ollama_ms = 14.0; // Ollama baseline
    let gap_ms = actual_ms - ollama_ms;
    println!("\n=== Impact Analysis ===");
    println!("Current gap to Ollama: {:.1} ms", gap_ms);
    println!(
        "Q8K overhead: {:.2} ms ({:.0}% of gap)",
        overhead_per_token_ms,
        overhead_per_token_ms / gap_ms * 100.0
    );

    // Could we skip Q8K?
    // The Q8K path gives VNNI acceleration but adds quantization overhead
    // Let's see if the net benefit is positive

    Ok(())
}
