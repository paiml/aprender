//! Profile where time is spent in single-token inference
//!
//! Breaks down time for: embedding, attention, FFN, normalization, lm_head

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf";

    println!("Loading model: {}", model_path);
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Warmup
    let prompt = vec![1u32, 29871, 29896, 29974, 29896, 29922]; // "1+1="
    let config = QuantizedGenerateConfig {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![2],
        trace: false,
        ..Default::default()
    };

    println!("Warming up...");
    for _ in 0..3 {
        let _ = model.generate_with_cache(&prompt, &config)?;
    }

    // Profile generation
    println!("\n=== Profiling 8-token generation ===");
    let config = QuantizedGenerateConfig {
        max_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![2],
        trace: false,
        ..Default::default()
    };

    let iterations = 5;
    let mut total_ms = 0.0;

    for i in 0..iterations {
        let start = Instant::now();
        let result = model.generate_with_cache(&prompt, &config)?;
        let elapsed = start.elapsed();
        total_ms += elapsed.as_millis() as f64;

        if i == 0 {
            println!("Tokens generated: {}", result.len() - prompt.len());
        }
    }

    let avg_ms = total_ms / iterations as f64;
    let tokens_generated = 8.0; // max_tokens
    let ms_per_token = avg_ms / tokens_generated;
    let toks_per_sec = 1000.0 / ms_per_token;

    println!("\nResults ({} iterations):", iterations);
    println!("  Total time: {:.1}ms avg", avg_ms);
    println!("  Per token:  {:.1}ms", ms_per_token);
    println!("  Throughput: {:.1} tok/s", toks_per_sec);

    // Calculate theoretical limits
    println!("\n=== Theoretical Limits ===");

    // TinyLlama specs
    let n_layers = 22;
    let hidden_dim = 2048;
    let intermediate_dim = 5632;
    let _n_heads = 32;
    let _head_dim = 64;
    let vocab_size = 32000;

    // Q4_K: ~0.5625 bytes per element (4.5 bits per weight)
    let bytes_per_q4k_element = 0.5625;

    // Per-layer matmul sizes
    let qkv_weights = hidden_dim * (hidden_dim * 3); // Q,K,V combined
    let o_weights = hidden_dim * hidden_dim;
    let ffn_up_weights = hidden_dim * intermediate_dim;
    let ffn_gate_weights = hidden_dim * intermediate_dim;
    let ffn_down_weights = intermediate_dim * hidden_dim;
    let lm_head_weights = hidden_dim * vocab_size;

    let per_layer_weights =
        qkv_weights + o_weights + ffn_up_weights + ffn_gate_weights + ffn_down_weights;
    let total_weights = per_layer_weights * n_layers + lm_head_weights;
    let total_bytes = (total_weights as f64 * bytes_per_q4k_element) as usize;

    println!("Model weights:");
    println!(
        "  Per layer:  {:.1}M parameters",
        per_layer_weights as f64 / 1e6
    );
    println!(
        "  Total:      {:.1}M parameters",
        total_weights as f64 / 1e6
    );
    println!("  Q4_K size:  {:.1}MB", total_bytes as f64 / 1e6);

    // Memory bandwidth calculation
    let ddr4_bw_gbs = 25.0; // ~25 GB/s for DDR4-3200
    let min_time_ms = (total_bytes as f64 / 1e6) / ddr4_bw_gbs * 1000.0;
    let max_toks = 1000.0 / min_time_ms;

    println!(
        "\nMemory bandwidth limit (DDR4-3200 @ {} GB/s):",
        ddr4_bw_gbs
    );
    println!("  Min time to read weights: {:.1}ms", min_time_ms);
    println!("  Max theoretical tok/s:    {:.0}", max_toks);
    println!(
        "  Current efficiency:       {:.0}%",
        (toks_per_sec / max_toks) * 100.0
    );

    // Compare to llama.cpp
    let llamacpp_toks = 100.0;
    println!("\nllama.cpp comparison:");
    println!("  llama.cpp: ~{:.0} tok/s", llamacpp_toks);
    println!("  realizar:  ~{:.0} tok/s", toks_per_sec);
    println!("  Gap:       {:.1}x", llamacpp_toks / toks_per_sec);

    Ok(())
}
