//! Quick benchmark for Qwen2.5-Coder-1.5B
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
        max_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };
    let _ = model.generate_with_cache(&prompt, &config)?;

    // Benchmark
    let gen_tokens = 20;
    let iterations = 5;

    let config = QuantizedGenerateConfig {
        max_tokens: gen_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Cache path
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.generate_with_cache(&prompt, &config)?;
    }
    let cache_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    let cache_tps = gen_tokens as f64 * 1000.0 / cache_ms;

    // Scratch path
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.generate_with_scratch(&prompt, &config)?;
    }
    let scratch_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    let scratch_tps = gen_tokens as f64 * 1000.0 / scratch_ms;

    println!(
        "\n=== Qwen2.5-Coder-1.5B Benchmark ({} tokens, {} iterations) ===",
        gen_tokens, iterations
    );
    println!("Cache path:   {:.1} ms ({:.1} tok/s)", cache_ms, cache_tps);
    println!(
        "Scratch path: {:.1} ms ({:.1} tok/s)",
        scratch_ms, scratch_tps
    );
    println!();
    println!("Ollama baseline: 70.59 tok/s");
    println!("Our best:        {:.1} tok/s", scratch_tps.max(cache_tps));
    println!(
        "Gap:             {:.1}x slower",
        70.59 / scratch_tps.max(cache_tps)
    );
    println!("Target (2x):     142 tok/s");

    Ok(())
}
