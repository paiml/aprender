//! Profile individual operations in the forward pass
//!
//! Measures: matmul, attention, RMSNorm, RoPE, allocations

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.config().num_layers;
    let num_heads = model.config().num_heads;

    println!(
        "Model: {} layers, {} hidden, {} heads",
        num_layers, hidden_dim, num_heads
    );

    // Warm up with a few tokens
    let prompt = vec![1u32, 29871, 29896];
    let config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![2],
        trace: false,
        ..Default::default()
    };
    for _ in 0..3 {
        let _ = model.generate_with_cache(&prompt, &config)?;
    }

    // Now generate with longer context to stress attention
    println!("\n=== Profiling with growing KV cache ===");

    let max_tokens = 32;
    let _config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Track timing per token
    let mut times: Vec<f64> = Vec::new();

    let mut cache = OwnedQuantizedKVCache::new(num_layers, hidden_dim, 512);

    // Process prompt
    for (i, &token) in prompt.iter().enumerate() {
        let start = Instant::now();
        let _ = model.forward_single_with_cache(token, &mut cache, i)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // Generate tokens
    let mut last_token = prompt[prompt.len() - 1];
    for i in 0..max_tokens {
        let pos = prompt.len() + i;
        let start = Instant::now();
        let logits = model.forward_single_with_cache(last_token, &mut cache, pos)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);

        // Argmax for next token
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (idx, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        last_token = max_idx as u32;
    }

    println!("\n=== Time per token (ms) ===");
    for (i, &t) in times.iter().enumerate() {
        let cache_len = if i < prompt.len() {
            i
        } else {
            prompt.len() + (i - prompt.len())
        };
        println!("Token {:2}: {:6.2} ms (cache_len={})", i, t, cache_len);
    }

    // Calculate averages
    let prompt_avg = times[..prompt.len()].iter().sum::<f64>() / prompt.len() as f64;
    let gen_avg = times[prompt.len()..].iter().sum::<f64>() / max_tokens as f64;

    println!("\n=== Summary ===");
    println!("Prompt tokens (cache small): {:.2} ms avg", prompt_avg);
    println!("Generated tokens (cache grows): {:.2} ms avg", gen_avg);
    println!("Slowdown with growing cache: {:.2}x", gen_avg / prompt_avg);

    // Attention complexity analysis
    let avg_cache_len =
        (0..max_tokens).map(|i| prompt.len() + i).sum::<usize>() as f64 / max_tokens as f64;
    println!(
        "\nAverage cache length during generation: {:.1}",
        avg_cache_len
    );
    println!("Attention scales as O(n) with cache length");
    println!(
        "Total heads × layers = {} attention computations per token",
        num_heads * num_layers
    );

    // Memory allocation estimate
    let allocations_per_token = num_heads * num_layers; // scores vectors in attention
    println!(
        "\nEstimated Vec allocations per token: {}",
        allocations_per_token
    );

    Ok(())
}
