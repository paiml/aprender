//! CORRECTNESS-002: Debug CPU vs GPU divergence
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_cpu_gpu_divergence

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    // Load model for CPU
    eprintln!("Loading model for CPU from {}", model_path);
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    eprintln!("CPU model loaded in {:?}", start.elapsed());

    // Display model config
    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let vocab_size = cpu_model.config().vocab_size;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    eprintln!("\nModel config:");
    eprintln!("  hidden_dim: {}", hidden_dim);
    eprintln!("  num_layers: {}", num_layers);
    eprintln!("  num_heads: {}", num_heads);
    eprintln!("  num_kv_heads: {}", num_kv_heads);
    eprintln!("  vocab_size: {}", vocab_size);

    // Simple single-token test
    let test_token: u32 = 791; // "What" token
    eprintln!("\n=== Testing with token {} ===", test_token);

    // CPU path: embedding lookup
    let embedding_cpu = cpu_model.embed(&[test_token]);
    let embed_sum_cpu: f32 = embedding_cpu.iter().sum();
    let embed_rms_cpu: f32 =
        (embedding_cpu.iter().map(|x| x * x).sum::<f32>() / embedding_cpu.len() as f32).sqrt();
    eprintln!(
        "[CPU] Embedding sum={:.4}, rms={:.4}, first 5={:?}",
        embed_sum_cpu,
        embed_rms_cpu,
        &embedding_cpu[..5]
    );

    // CPU path: single forward with cache
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits_cpu = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;
    let cpu_sum: f32 = logits_cpu.iter().sum();
    let cpu_max = logits_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_argmax = logits_cpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] Logits sum={:.4}, max={:.4}, argmax={}",
        cpu_sum, cpu_max, cpu_argmax
    );

    // Load fresh model for GPU
    eprintln!("\nLoading model for GPU...");
    let start = Instant::now();
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    eprintln!("GPU model loaded in {:?}", start.elapsed());

    // GPU path: Must preload weights first for GPU-resident path
    eprintln!("Preloading weights to GPU...");
    let bytes_uploaded = cuda_model.preload_weights_gpu()?;
    eprintln!(
        "Uploaded {} MB of weights to GPU",
        bytes_uploaded / (1024 * 1024)
    );

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    // Enable GPU debug output
    std::env::set_var("GPU_DEBUG", "1");

    let logits_gpu = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;
    let gpu_sum: f32 = logits_gpu.iter().sum();
    let gpu_max = logits_gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_argmax = logits_gpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[GPU] Logits sum={:.4}, max={:.4}, argmax={}",
        gpu_sum, gpu_max, gpu_argmax
    );

    // Compare top-5 tokens
    let mut cpu_sorted: Vec<(usize, f32)> = logits_cpu
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    cpu_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut gpu_sorted: Vec<(usize, f32)> = logits_gpu
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    gpu_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n=== Top 5 Comparison ===");
    eprintln!("CPU top 5: {:?}", &cpu_sorted[..5]);
    eprintln!("GPU top 5: {:?}", &gpu_sorted[..5]);

    // Cross-comparison: what does GPU say about CPU's top tokens?
    eprintln!("\n=== Cross-comparison ===");
    eprintln!("What GPU says about CPU's top 5 tokens:");
    for &(idx, cpu_val) in &cpu_sorted[..5] {
        let gpu_val = logits_gpu[idx];
        eprintln!(
            "  Token {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
            idx,
            cpu_val,
            gpu_val,
            cpu_val - gpu_val
        );
    }

    eprintln!("\nWhat CPU says about GPU's top 5 tokens:");
    for &(idx, gpu_val) in &gpu_sorted[..5] {
        let cpu_val = logits_cpu[idx];
        eprintln!(
            "  Token {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
            idx,
            cpu_val,
            gpu_val,
            cpu_val - gpu_val
        );
    }

    // Check if they match
    let match_count = cpu_sorted[..5]
        .iter()
        .zip(gpu_sorted[..5].iter())
        .filter(|(c, g)| c.0 == g.0)
        .count();

    if match_count == 5 {
        eprintln!("\n[OK] CPU and GPU agree on top 5 tokens");
    } else {
        eprintln!("\n[MISMATCH] Only {}/5 top tokens match", match_count);
        eprintln!("[FAIL] CORRECTNESS-002 confirmed: GPU produces different output than CPU");
    }

    Ok(())
}
