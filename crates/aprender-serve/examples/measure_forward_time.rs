//! Measure actual forward pass time for Qwen2.5-Coder-1.5B
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    eprintln!("Loading model: {}", model_path);
    let mapped = MappedGGUFModel::from_path(&model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    eprintln!(
        "Model: hidden_dim={}, intermediate_dim={}, layers={}",
        model.config().hidden_dim,
        model.config().intermediate_dim,
        model.config().num_layers
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
    let _ = model.generate_with_cache(&prompt, &config)?;
    let _ = model.generate_with_scratch(&prompt, &config)?;

    // Measure forward pass only (no sampling overhead)
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 50, // Generate 50 tokens
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    let iterations = 3;

    // Cache path
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.generate_with_cache(&prompt, &gen_config)?;
    }
    let cache_total_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    let cache_per_token_ms = cache_total_ms / 50.0;

    // Scratch path
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.generate_with_scratch(&prompt, &gen_config)?;
    }
    let scratch_total_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    let scratch_per_token_ms = scratch_total_ms / 50.0;

    println!(
        "\n=== Forward Pass Timing (50 tokens, {} iterations) ===",
        iterations
    );
    println!("Cache path:");
    println!("  Total:     {:.1} ms", cache_total_ms);
    println!(
        "  Per token: {:.1} ms ({:.1} tok/s)",
        cache_per_token_ms,
        1000.0 / cache_per_token_ms
    );
    println!("Scratch path:");
    println!("  Total:     {:.1} ms", scratch_total_ms);
    println!(
        "  Per token: {:.1} ms ({:.1} tok/s)",
        scratch_per_token_ms,
        1000.0 / scratch_per_token_ms
    );

    // Compare to estimated
    let estimated_per_token_ms = 46.8; // From profile
    println!("\n=== Gap Analysis ===");
    println!("Estimated per token: {:.1} ms", estimated_per_token_ms);
    println!("Actual (cache):      {:.1} ms", cache_per_token_ms);
    println!("Actual (scratch):    {:.1} ms", scratch_per_token_ms);
    println!(
        "Gap (cache):         {:.1} ms ({:.0}% overhead)",
        cache_per_token_ms - estimated_per_token_ms,
        (cache_per_token_ms / estimated_per_token_ms - 1.0) * 100.0
    );
    println!(
        "Gap (scratch):       {:.1} ms ({:.0}% overhead)",
        scratch_per_token_ms - estimated_per_token_ms,
        (scratch_per_token_ms / estimated_per_token_ms - 1.0) * 100.0
    );

    Ok(())
}
