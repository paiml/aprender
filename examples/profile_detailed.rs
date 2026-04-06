//! Detailed profiling of transformer forward pass
//!
//! Measures time breakdown: embedding, attention, FFN, normalization, lm_head

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf";

    println!("Loading model: {}", model_path);
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Get model dimensions
    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let n_layers = model.config().num_layers;
    let vocab_size = model.config().vocab_size;

    println!("\nModel Architecture:");
    println!("  Hidden dim:       {}", hidden_dim);
    println!("  Intermediate dim: {}", intermediate_dim);
    println!("  Layers:           {}", n_layers);
    println!("  Vocab size:       {}", vocab_size);

    // Calculate bytes per layer
    let q4k_bytes_per_param = 0.5625; // 4.5 bits per weight

    // Per-layer matmul dimensions
    let qkv_params = hidden_dim * (hidden_dim * 3); // Q,K,V combined
    let o_params = hidden_dim * hidden_dim;
    let ffn_up_params = hidden_dim * intermediate_dim;
    let ffn_gate_params = hidden_dim * intermediate_dim;
    let ffn_down_params = intermediate_dim * hidden_dim;

    let attn_params = qkv_params + o_params;
    let ffn_params = ffn_up_params + ffn_gate_params + ffn_down_params;
    let lm_head_params = hidden_dim * vocab_size;

    let attn_bytes = (attn_params as f64 * q4k_bytes_per_param) as usize;
    let ffn_bytes = (ffn_params as f64 * q4k_bytes_per_param) as usize;
    let lm_head_bytes = (lm_head_params as f64 * q4k_bytes_per_param) as usize;

    println!("\nMemory per layer:");
    println!(
        "  Attention weights: {:.2} MB ({} params)",
        attn_bytes as f64 / 1e6,
        attn_params
    );
    println!(
        "  FFN weights:       {:.2} MB ({} params)",
        ffn_bytes as f64 / 1e6,
        ffn_params
    );
    println!(
        "  LM head:           {:.2} MB ({} params)",
        lm_head_bytes as f64 / 1e6,
        lm_head_params
    );

    let total_bytes = (attn_bytes + ffn_bytes) * n_layers + lm_head_bytes;
    println!("\nTotal model weights: {:.2} MB", total_bytes as f64 / 1e6);

    // Run multiple tokens to measure steady-state performance
    let prompt = vec![1u32, 29871, 29896, 29974, 29896, 29922]; // "1+1="
    let config = QuantizedGenerateConfig {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![2],
        trace: false,
        ..Default::default()
    };

    // Warmup
    println!("\nWarming up (3 iterations)...");
    for _ in 0..3 {
        let _ = model.generate_with_scratch(&prompt, &config)?;
    }

    // Benchmark multiple runs
    let iterations = 10;
    let mut times_ms: Vec<f64> = Vec::new();

    println!("\nBenchmarking ({} iterations)...", iterations);
    for i in 0..iterations {
        let start = Instant::now();
        let tokens = model.generate_with_scratch(&prompt, &config)?;
        let elapsed = start.elapsed();

        let generated = tokens.len() - prompt.len();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let ms_per_tok = ms / generated as f64;
        times_ms.push(ms_per_tok);

        if i == 0 {
            println!("  Generated {} tokens", generated);
        }
    }

    // Calculate statistics
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times_ms[0];
    let max = times_ms[times_ms.len() - 1];
    let median = times_ms[times_ms.len() / 2];
    let mean: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;

    println!("\n=== Performance Results ===");
    println!("Time per token:");
    println!("  Min:    {:.1} ms ({:.1} tok/s)", min, 1000.0 / min);
    println!("  Median: {:.1} ms ({:.1} tok/s)", median, 1000.0 / median);
    println!("  Mean:   {:.1} ms ({:.1} tok/s)", mean, 1000.0 / mean);
    println!("  Max:    {:.1} ms ({:.1} tok/s)", max, 1000.0 / max);

    // Memory bandwidth analysis
    println!("\n=== Memory Bandwidth Analysis ===");
    let ddr4_bw_gbs = 25.0; // DDR4-3200 theoretical peak
    let ddr5_bw_gbs = 60.0; // DDR5-4800 theoretical peak

    // Effective bandwidth = bytes read / time
    let effective_bw_gbs = (total_bytes as f64 / 1e9) / (median / 1000.0);
    println!("Effective bandwidth: {:.1} GB/s", effective_bw_gbs);
    println!(
        "DDR4-3200 peak:      {:.1} GB/s ({:.0}% utilized)",
        ddr4_bw_gbs,
        (effective_bw_gbs / ddr4_bw_gbs) * 100.0
    );
    println!(
        "DDR5-4800 peak:      {:.1} GB/s ({:.0}% utilized)",
        ddr5_bw_gbs,
        (effective_bw_gbs / ddr5_bw_gbs) * 100.0
    );

    // Theoretical limits
    let min_time_ddr4 = (total_bytes as f64 / 1e9) / ddr4_bw_gbs * 1000.0;
    let min_time_ddr5 = (total_bytes as f64 / 1e9) / ddr5_bw_gbs * 1000.0;
    println!("\nTheoretical minimum time per token:");
    println!(
        "  DDR4: {:.1} ms ({:.0} tok/s max)",
        min_time_ddr4,
        1000.0 / min_time_ddr4
    );
    println!(
        "  DDR5: {:.1} ms ({:.0} tok/s max)",
        min_time_ddr5,
        1000.0 / min_time_ddr5
    );

    // llama.cpp comparison
    let llamacpp_toks = 100.0;
    let gap = llamacpp_toks / (1000.0 / median);
    println!("\n=== llama.cpp Comparison ===");
    println!("llama.cpp: ~{:.0} tok/s (reported baseline)", llamacpp_toks);
    println!("realizar:  ~{:.1} tok/s", 1000.0 / median);
    println!("Gap:       {:.1}x slower", gap);

    // Breakdown of what's needed to close the gap
    println!("\n=== Gap Analysis ===");
    if gap > 3.0 {
        println!("Performance gap suggests:");
        println!(
            "  1. Memory bandwidth not saturated ({:.0}% DDR4)",
            (effective_bw_gbs / ddr4_bw_gbs) * 100.0
        );
        println!("  2. Possible CPU stalls (branch mispredicts, cache misses)");
        println!("  3. Missing optimizations (tiled matmul, Q8 activations)");
    } else if gap > 1.5 {
        println!("Performance gap suggests:");
        println!("  1. Single-threaded vs multi-threaded difference");
        println!("  2. Missing L3 cache optimizations");
    } else {
        println!("Within acceptable range of llama.cpp performance!");
    }

    // Architecture-specific recommendations
    println!("\n=== Optimization Recommendations ===");
    if effective_bw_gbs < ddr4_bw_gbs * 0.7 {
        println!("1. Not memory-bound yet - CPU is the bottleneck");
        println!("   → Profile CPU hotspots (perf record / flamegraph)");
    } else {
        println!("1. Memory-bound - need to reduce traffic");
        println!("   → Use Q8_0 activations (3.5x less activation bandwidth)");
        println!("   → Implement cache-blocked matmul");
    }

    Ok(())
}
