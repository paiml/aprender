//! Benchmark: generate_with_cache vs generate_with_scratch
//!
//! Tests correctness and performance of the zero-allocation inference path.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf";

    println!("Loading model: {}", model_path);
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let owned_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Test prompt: "1+1=" = [1, 29871, 29896, 29974, 29896, 29922]
    let prompt = vec![1u32, 29871, 29896, 29974, 29896, 29922];
    println!("Prompt tokens: {:?}", prompt);

    let config = QuantizedGenerateConfig {
        max_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![2], // EOS
        trace: false,
        ..Default::default()
    };

    // Test correctness: both should produce identical output with greedy decoding
    println!("\n=== CORRECTNESS TEST ===");
    let result_cache = owned_model.generate_with_cache(&prompt, &config)?;
    let result_scratch = owned_model.generate_with_scratch(&prompt, &config)?;

    println!("generate_with_cache:   {:?}", result_cache);
    println!("generate_with_scratch: {:?}", result_scratch);

    if result_cache == result_scratch {
        println!("✓ Results match!");
    } else {
        println!("✗ Results differ!");
        // Compare token by token
        for (i, (a, b)) in result_cache.iter().zip(result_scratch.iter()).enumerate() {
            if a != b {
                println!("  Position {}: cache={}, scratch={}", i, a, b);
            }
        }
        return Err(RealizarError::InvalidShape {
            reason: "generate_with_scratch produced different output".into(),
        });
    }

    // Benchmark
    println!("\n=== BENCHMARK ===");
    let iterations = 10;

    // Warmup
    for _ in 0..3 {
        let _ = owned_model.generate_with_cache(&prompt, &config)?;
        let _ = owned_model.generate_with_scratch(&prompt, &config)?;
    }

    // Benchmark generate_with_cache
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = owned_model.generate_with_cache(&prompt, &config)?;
    }
    let cache_total = start.elapsed();
    let cache_avg_ms = cache_total.as_millis() as f64 / iterations as f64;

    // Benchmark generate_with_scratch
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = owned_model.generate_with_scratch(&prompt, &config)?;
    }
    let scratch_total = start.elapsed();
    let scratch_avg_ms = scratch_total.as_millis() as f64 / iterations as f64;

    // Results
    let speedup = cache_avg_ms / scratch_avg_ms;
    println!("generate_with_cache:   {:.1}ms avg", cache_avg_ms);
    println!("generate_with_scratch: {:.1}ms avg", scratch_avg_ms);
    println!("Speedup: {:.2}x", speedup);

    // Tokens per second
    let tokens_generated = (result_cache.len() - prompt.len()) as f64;
    let toks_cache = tokens_generated / (cache_avg_ms / 1000.0);
    let toks_scratch = tokens_generated / (scratch_avg_ms / 1000.0);
    println!("\nTokens/second:");
    println!("  generate_with_cache:   {:.1} tok/s", toks_cache);
    println!("  generate_with_scratch: {:.1} tok/s", toks_scratch);

    Ok(())
}
