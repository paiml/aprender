//! PAR-126: Detailed forward pass profiler
//! Goal: Find where the 13.5 ms unexplained overhead comes from

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== PAR-126 Detailed Forward Pass Profiler ===\n");
    println!(
        "Model: {} layers, hidden={}, intermediate={}, heads={}, kv_heads={}",
        model.config().num_layers,
        model.config().hidden_dim,
        model.config().intermediate_dim,
        model.config().num_heads,
        model.config().num_kv_heads
    );

    // Warmup
    let prompt = vec![1u32, 2, 3, 4];
    let config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };
    let _ = model.generate_with_scratch(&prompt, &config)?;

    // Detailed timing - generate N tokens and measure each stage
    let tokens_to_generate = 20;
    let gen_config = QuantizedGenerateConfig {
        max_tokens: tokens_to_generate,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Run multiple times to get stable measurements
    let iterations = 5;
    let mut total_times_ms = Vec::new();

    for _iter in 0..iterations {
        let start = Instant::now();
        let _ = model.generate_with_scratch(&prompt, &gen_config)?;
        let total_ms = start.elapsed().as_micros() as f64 / 1000.0;
        total_times_ms.push(total_ms);
    }

    // Calculate stats
    let avg_total_ms: f64 = total_times_ms.iter().sum::<f64>() / iterations as f64;
    let per_token_ms = avg_total_ms / tokens_to_generate as f64;
    let tok_per_s = 1000.0 / per_token_ms;

    println!("\n=== Timing Summary ===");
    println!("Tokens generated: {}", tokens_to_generate);
    println!("Average total:    {:.1} ms", avg_total_ms);
    println!("Per token:        {:.2} ms", per_token_ms);
    println!("Throughput:       {:.1} tok/s", tok_per_s);

    // Calculate expected breakdown
    let num_layers = model.config().num_layers;
    let _hidden_dim = model.config().hidden_dim;
    let _intermediate_dim = model.config().intermediate_dim;

    // Based on profiling data from v4.94.0:
    // - Per-layer matmul: 577 µs (Q4_K + Q6_K SIMD)
    // - Non-matmul ops: ~65 µs/layer
    let matmul_per_layer_us = 577.0;
    let non_matmul_per_layer_us = 65.0;
    let total_per_layer_us = matmul_per_layer_us + non_matmul_per_layer_us;
    let expected_per_token_ms = total_per_layer_us * num_layers as f64 / 1000.0;

    println!("\n=== Expected vs Actual ===");
    println!(
        "Expected (matmul + ops): {:.1} ms/token",
        expected_per_token_ms
    );
    println!("Actual:                  {:.1} ms/token", per_token_ms);
    println!(
        "Unexplained overhead:    {:.1} ms/token ({:.0}%)",
        per_token_ms - expected_per_token_ms,
        (per_token_ms - expected_per_token_ms) / per_token_ms * 100.0
    );

    // Measure Rayon dispatch overhead
    println!("\n=== Rayon Dispatch Overhead Test ===");
    let dispatch_iterations = 1000;
    let start = Instant::now();
    for _ in 0..dispatch_iterations {
        rayon::join(|| 1, || 2);
    }
    let dispatch_us = start.elapsed().as_micros() as f64 / dispatch_iterations as f64;
    println!("rayon::join overhead: {:.1} µs per call", dispatch_us);

    // Per-token Rayon calls estimate
    // - 1 join for FFN up/gate per layer
    // - Multiple par_iter calls for matmuls
    let matmuls_per_layer = 5; // QKV, attn_out, up, gate, down
    let rayon_calls_per_layer = 1 + matmuls_per_layer; // join + matmuls
    let rayon_calls_per_token = rayon_calls_per_layer * num_layers;
    let rayon_overhead_ms = dispatch_us * rayon_calls_per_token as f64 / 1000.0;
    println!(
        "Rayon calls per token: {} ({} layers × {} calls/layer)",
        rayon_calls_per_token, num_layers, rayon_calls_per_layer
    );
    println!(
        "Estimated Rayon overhead: {:.1} ms/token",
        rayon_overhead_ms
    );

    // Memory bandwidth test
    println!("\n=== Memory Bandwidth Test ===");
    let data_size = 100 * 1024 * 1024; // 100 MB
    let data: Vec<f32> = vec![1.0; data_size / 4];
    let mut sum = 0.0f32;

    let start = Instant::now();
    for &v in &data {
        sum += v;
    }
    let read_time_ms = start.elapsed().as_micros() as f64 / 1000.0;
    let bw_gb_s = (data_size as f64 / (1024.0 * 1024.0 * 1024.0)) / (read_time_ms / 1000.0);
    println!(
        "Sequential read 100MB: {:.1} ms ({:.1} GB/s)",
        read_time_ms, bw_gb_s
    );
    println!("(sum = {} to prevent optimization)", sum); // Prevent dead code elimination

    // Comparison
    println!("\n=== Ollama Comparison ===");
    let ollama_ms = 14.05;
    println!("Ollama:       {:.2} ms/tok (71.2 tok/s)", ollama_ms);
    println!(
        "realizar:     {:.2} ms/tok ({:.1} tok/s)",
        per_token_ms, tok_per_s
    );
    println!("Gap:          {:.2}x", per_token_ms / ollama_ms);

    // What would it take to match Ollama?
    let overhead_to_remove = per_token_ms - ollama_ms;
    println!("\n=== Path to Parity ===");
    println!("Overhead to remove: {:.1} ms/token", overhead_to_remove);
    println!(
        "If Rayon overhead ({:.1} ms) removed: {:.1} ms/tok ({:.1} tok/s)",
        rayon_overhead_ms,
        per_token_ms - rayon_overhead_ms,
        1000.0 / (per_token_ms - rayon_overhead_ms)
    );

    Ok(())
}
