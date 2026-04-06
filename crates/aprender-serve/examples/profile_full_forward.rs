//! PAR-126: Profile ALL operations in a single forward pass

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map_or(
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        |s| s.as_str(),
    );

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!(
        "Model: {} layers, hidden={}, intermediate={}",
        model.config().num_layers,
        model.config().hidden_dim,
        model.config().intermediate_dim
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

    // Profile generate
    let iterations = 5;
    let tokens_per_iter = 50;

    let gen_config = QuantizedGenerateConfig {
        max_tokens: tokens_per_iter,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Scratch path
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.generate_with_scratch(&prompt, &gen_config)?;
    }
    let total_ms = start.elapsed().as_millis() as f64;
    let per_token_ms = total_ms / (iterations * tokens_per_iter) as f64;
    let tok_per_sec = 1000.0 / per_token_ms;

    println!("\n=== Performance Summary ===");
    println!("Per token:    {:.2} ms", per_token_ms);
    println!("Throughput:   {:.1} tok/s", tok_per_sec);

    // Compare to Ollama
    let ollama_tok_s = 71.17;
    let ollama_ms = 1000.0 / ollama_tok_s;
    println!("\n=== vs Ollama (CPU) ===");
    println!(
        "Ollama:       {:.2} ms/tok ({:.1} tok/s)",
        ollama_ms, ollama_tok_s
    );
    println!(
        "realizar:     {:.2} ms/tok ({:.1} tok/s)",
        per_token_ms, tok_per_sec
    );
    println!("Gap:          {:.2}x", per_token_ms / ollama_ms);

    // What we need
    let target_2x = ollama_tok_s * 2.0;
    let target_ms = 1000.0 / target_2x;
    println!("\n=== Target: 2x Ollama ===");
    println!(
        "Target:       {:.2} ms/tok ({:.1} tok/s)",
        target_ms, target_2x
    );
    println!(
        "Gap:          {:.2}x speedup needed",
        per_token_ms / target_ms
    );

    Ok(())
}
