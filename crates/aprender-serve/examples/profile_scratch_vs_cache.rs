//! Micro-benchmark: compare scratch vs cache path performance
//! Identifies exact source of overhead
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/noah/models/Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf".to_string());

    eprintln!("Loading model: {}", model_path);
    let mapped = MappedGGUFModel::from_path(&model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let num_layers = model.config().num_layers;

    eprintln!(
        "Model: hidden_dim={}, intermediate_dim={}, layers={}",
        hidden_dim, intermediate_dim, num_layers
    );

    // Warmup
    let prompt = vec![1u32, 2, 3, 4, 5];
    let config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };
    let _ = model.generate_with_cache(&prompt, &config)?;

    // Test 1: Generate with cache path (allocating)
    let iterations = 5;
    let gen_tokens = 10;

    let config = QuantizedGenerateConfig {
        max_tokens: gen_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.generate_with_cache(&prompt, &config)?;
    }
    let cache_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    let cache_tok_per_s = gen_tokens as f64 * 1000.0 / cache_ms;

    // Test 2: Generate with scratch path (non-allocating)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.generate_with_scratch(&prompt, &config)?;
    }
    let scratch_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    let scratch_tok_per_s = gen_tokens as f64 * 1000.0 / scratch_ms;

    println!(
        "\n=== Results ({} tokens, {} iterations) ===",
        gen_tokens, iterations
    );
    println!(
        "Cache path (allocating):     {:.1} ms ({:.1} tok/s)",
        cache_ms, cache_tok_per_s
    );
    println!(
        "Scratch path (non-alloc):    {:.1} ms ({:.1} tok/s)",
        scratch_ms, scratch_tok_per_s
    );
    println!(
        "Overhead:                    {:.1} ms ({:.1}%)",
        scratch_ms - cache_ms,
        (scratch_ms / cache_ms - 1.0) * 100.0
    );

    // Test 3: Per-token timing derived from generation
    let per_token_cache_us = cache_ms * 1000.0 / gen_tokens as f64;
    let per_token_scratch_us = scratch_ms * 1000.0 / gen_tokens as f64;

    println!("\n=== Per-Token Timing (derived) ===");
    println!("Per token (cache):           {:.0} us", per_token_cache_us);
    println!(
        "Per token (scratch):         {:.0} us",
        per_token_scratch_us
    );
    println!(
        "Gap per token:               {:.0} us",
        per_token_scratch_us - per_token_cache_us
    );

    // Analysis
    println!("\n=== Analysis ===");
    if scratch_ms > cache_ms {
        println!("UNEXPECTED: Scratch path is SLOWER than cache path!");
        println!("Possible causes:");
        println!("  1. Extra memory copies in scratch path");
        println!("  2. Cache path has more optimized code path");
        println!("  3. Different loop structure overhead");
    } else {
        println!(
            "Scratch path is {:.1}% faster",
            (cache_ms / scratch_ms - 1.0) * 100.0
        );
    }

    Ok(())
}
