//! CORRECTNESS-002: Compare normed_hidden states between CPU and GPU
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_normed_hidden

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Get config
    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let vocab_size = cpu_model.config().vocab_size;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let test_token: u32 = 791;
    eprintln!("\n=== Testing token {} ===", test_token);

    // CPU: Run forward and capture internal states
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    // Get embedding
    let embedding = cpu_model.embed(&[test_token]);
    eprintln!("[CPU] Embedding: first 5={:?}", &embedding[..5]);

    // Run through transformer layers (we need to access internal forward method)
    // For now, just use the full forward and compare final output
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;
    let cpu_sum: f32 = cpu_logits.iter().sum();
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] Logits: sum={:.2}, argmax={}, first 5={:?}",
        cpu_sum,
        cpu_argmax,
        &cpu_logits[..5]
    );

    // GPU: Run forward with debug output
    eprintln!("\nLoading GPU model...");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // Enable debug
    std::env::set_var("GPU_DEBUG", "1");

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;
    let gpu_sum: f32 = gpu_logits.iter().sum();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[GPU] Logits: sum={:.2}, argmax={}, first 5={:?}",
        gpu_sum,
        gpu_argmax,
        &gpu_logits[..5]
    );

    // Now test with the SAME normed_hidden input to LM head
    // Use GPU's normed_hidden values from above as input to CPU's Q6K matmul
    eprintln!("\n=== Testing LM head with identical input ===");

    // Synthetic test: use all-ones input to LM head
    let ones_input: Vec<f32> = vec![1.0; hidden_dim];

    // CPU LM head with ones input
    let cpu_lm_logits_ones = realizar::quantize::fused_q6k_parallel_matvec(
        &cpu_model.lm_head_weight().data,
        &ones_input,
        hidden_dim,
        vocab_size,
    )?;
    let cpu_ones_sum: f32 = cpu_lm_logits_ones.iter().sum();
    let cpu_ones_argmax = cpu_lm_logits_ones
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU LM head with ones] sum={:.2}, argmax={}, first 5={:?}",
        cpu_ones_sum,
        cpu_ones_argmax,
        &cpu_lm_logits_ones[..5]
    );

    // GPU LM head with ones input - directly test the Q6K kernel
    // We can't easily call just the LM head, but we can compare the results

    // Element-wise logit comparison
    eprintln!("\n=== Element-wise Comparison ===");
    let mut dot = 0.0f64;
    let mut cpu_sq = 0.0f64;
    let mut gpu_sq = 0.0f64;
    for i in 0..vocab_size {
        let c = cpu_logits[i] as f64;
        let g = gpu_logits[i] as f64;
        dot += c * g;
        cpu_sq += c * c;
        gpu_sq += g * g;
    }
    let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
    eprintln!("Correlation: {:.6}", corr);

    if corr > 0.99 {
        eprintln!("\n[OK] GPU matches CPU");
    } else if corr < 0.0 {
        eprintln!("\n[FAIL] Negative correlation between CPU and GPU logits");
    } else {
        eprintln!("\n[FAIL] GPU diverges from CPU (corr={:.4})", corr);
    }

    Ok(())
}
