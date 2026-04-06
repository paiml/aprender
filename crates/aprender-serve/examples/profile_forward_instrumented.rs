//! PAR-126: Fully instrumented forward pass profiler
//! Measures time between every operation to find hidden overhead

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.config().num_layers;
    let iterations = 1000usize;

    println!("=== PAR-126 Instrumented Forward Pass ===\n");
    println!("Model: {} layers, hidden={}", num_layers, hidden_dim);

    // Test 1: Measure overhead of scratch buffer creation
    let start = Instant::now();
    for _ in 0..100 {
        let _cache = realizar::gguf::OwnedQuantizedKVCache::from_config(model.config(), 100);
    }
    let cache_create_us = start.elapsed().as_micros() as f64 / 100.0;
    println!("KV Cache creation: {:.1} µs", cache_create_us);

    let start = Instant::now();
    for _ in 0..100 {
        let _scratch = realizar::gguf::InferenceScratchBuffer::from_config(model.config());
    }
    let scratch_create_us = start.elapsed().as_micros() as f64 / 100.0;
    println!("Scratch buffer creation: {:.1} µs", scratch_create_us);

    // Test 2: Measure cache append overhead
    let mut cache = realizar::gguf::OwnedQuantizedKVCache::from_config(model.config(), 1000);
    let kv_dim = model.config().num_kv_heads * (hidden_dim / model.config().num_heads);
    let k_data = vec![0.1f32; kv_dim];
    let v_data = vec![0.1f32; kv_dim];

    let start = Instant::now();
    for layer_idx in 0..num_layers {
        for _ in 0..100 {
            cache.append(layer_idx, &k_data, &v_data);
        }
    }
    // Reset by recreating
    let _ = cache;
    let append_us = start.elapsed().as_micros() as f64 / (100.0 * num_layers as f64);
    println!("Cache append: {:.1} µs per layer", append_us);

    // Measure standalone operations using implementations matching the actual code

    // Test 3: Measure RMSNorm (standalone)
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| i as f32 / hidden_dim as f32)
        .collect();
    let weight: Vec<f32> = vec![1.0f32; hidden_dim];
    let mut normed = vec![0.0f32; hidden_dim];
    let eps = 1e-5f32;

    let start = Instant::now();
    for _ in 0..iterations {
        rms_norm_into(&input, &weight, eps, &mut normed);
    }
    let rmsnorm_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("RMSNorm: {:.1} µs", rmsnorm_us);

    // Test 4: Measure RoPE (standalone)
    let mut q = vec![0.1f32; hidden_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        apply_rope(&mut q, 50, model.config().num_heads);
    }
    let rope_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("RoPE: {:.1} µs", rope_us);

    // Test 5: Measure copy_from_slice (QKV splitting)
    let qkv = vec![0.1f32; hidden_dim * 2]; // Approximate QKV dim
    let mut q_buf = vec![0.0f32; hidden_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        q_buf.copy_from_slice(&qkv[..hidden_dim]);
    }
    let copy_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("copy_from_slice (hidden): {:.1} µs", copy_us);

    // Test 6: Measure residual add
    let mut hidden_buf = vec![0.1f32; hidden_dim];
    let residual = vec![0.1f32; hidden_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..hidden_dim {
            hidden_buf[i] += residual[i];
        }
    }
    let add_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Residual add: {:.1} µs", add_us);

    // Test 7: Measure embedding lookup (standalone)
    // Just copying from embedding table
    let embed_table = vec![0.1f32; 151936 * hidden_dim / 8]; // Simplified
    let mut embed_out = vec![0.0f32; hidden_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        // Simulate embedding lookup - just copy operation
        let offset = 100 * hidden_dim;
        if offset + hidden_dim <= embed_table.len() {
            embed_out.copy_from_slice(&embed_table[offset..offset + hidden_dim]);
        }
    }
    let embed_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Embedding lookup: {:.1} µs", embed_us);

    // Summary: Per-layer overhead (excluding matmuls)
    println!("\n=== Per-Layer Overhead Summary ===");
    let per_layer_overhead = 2.0 * rmsnorm_us  // 2 RMSNorms
        + rope_us                               // RoPE for Q and K
        + 3.0 * copy_us                         // Q, K, V splitting
        + 2.0 * add_us                          // 2 residual adds
        + append_us; // Cache append

    let per_token_overhead_ms = per_layer_overhead * num_layers as f64 / 1000.0;
    println!("Per-layer overhead: {:.1} µs", per_layer_overhead);
    println!(
        "Per-token overhead (28 layers): {:.2} ms",
        per_token_overhead_ms
    );

    // Run actual forward pass for comparison
    println!("\n=== Actual Performance ===");
    let prompt = vec![1u32, 2, 3, 4];
    let config = QuantizedGenerateConfig {
        max_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Warmup
    let _ = model.generate_with_scratch(&prompt, &config)?;

    let runs = 5;
    let mut times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let _ = model.generate_with_scratch(&prompt, &config)?;
        times.push(start.elapsed().as_micros() as f64 / 1000.0);
    }

    let avg_ms = times.iter().sum::<f64>() / runs as f64;
    let per_token_ms = avg_ms / 20.0;
    println!("Average for 20 tokens: {:.1} ms", avg_ms);
    println!(
        "Per-token: {:.2} ms ({:.1} tok/s)",
        per_token_ms,
        1000.0 / per_token_ms
    );

    // Gap analysis
    println!("\n=== Gap Analysis ===");
    let matmul_ms = 14.7; // From profile_all_matmuls
    let accounted_ms = matmul_ms + per_token_overhead_ms;
    let unexplained_ms = per_token_ms - accounted_ms;
    println!("Matmuls: {:.1} ms", matmul_ms);
    println!("Non-matmul overhead: {:.2} ms", per_token_overhead_ms);
    println!("Accounted: {:.2} ms", accounted_ms);
    println!("Actual: {:.2} ms", per_token_ms);
    println!(
        "Unexplained: {:.2} ms ({:.0}%)",
        unexplained_ms,
        unexplained_ms / per_token_ms * 100.0
    );

    Ok(())
}

fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / input.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..input.len() {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

fn apply_rope(x: &mut [f32], position: usize, num_heads: usize) {
    let head_dim = x.len() / num_heads;
    let rope_base = 10000.0f32;
    for head in 0..num_heads {
        let head_offset = head * head_dim;
        for i in 0..head_dim / 2 {
            let freq = 1.0 / rope_base.powf(2.0 * i as f32 / head_dim as f32);
            let theta = position as f32 * freq;
            let (sin_t, cos_t) = theta.sin_cos();
            let re = x[head_offset + i];
            let im = x[head_offset + i + head_dim / 2];
            x[head_offset + i] = re * cos_t - im * sin_t;
            x[head_offset + i + head_dim / 2] = re * sin_t + im * cos_t;
        }
    }
}
