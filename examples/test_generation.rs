//! Test GPU generation produces coherent output

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let vocab_size = cpu_model.config().vocab_size;

    // Test tokens for "2+2="
    let tokens = vec![17, 10, 17, 28]; // Simple test

    eprintln!("\n=== CPU Generation ===");
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let mut cpu_last_logits = vec![];
    for (pos, &token) in tokens.iter().enumerate() {
        cpu_last_logits = cpu_model.forward_single_with_cache(token, &mut cpu_cache, pos)?;
    }
    let cpu_next = cpu_last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "CPU next token: {} (logit: {:.4})",
        cpu_next, cpu_last_logits[cpu_next]
    );

    eprintln!("\n=== GPU Generation ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let mut gpu_last_logits = vec![];
    for (pos, &token) in tokens.iter().enumerate() {
        gpu_last_logits = cuda_model.forward_gpu_resident(token, &mut gpu_cache, pos)?;
    }
    let gpu_next = gpu_last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "GPU next token: {} (logit: {:.4})",
        gpu_next, gpu_last_logits[gpu_next]
    );

    eprintln!("\n=== Comparison ===");
    if cpu_next == gpu_next {
        eprintln!("✅ CPU and GPU agree: token {}", cpu_next);
    } else {
        eprintln!("❌ Different: CPU={}, GPU={}", cpu_next, gpu_next);
        eprintln!("CPU logit at GPU choice: {:.4}", cpu_last_logits[gpu_next]);
        eprintln!("GPU logit at CPU choice: {:.4}", gpu_last_logits[cpu_next]);
    }

    // Calculate correlation
    let n = vocab_size;
    let cpu_mean: f32 = cpu_last_logits.iter().sum::<f32>() / n as f32;
    let gpu_mean: f32 = gpu_last_logits.iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut cpu_var = 0.0f32;
    let mut gpu_var = 0.0f32;
    for i in 0..n {
        let cpu_d = cpu_last_logits[i] - cpu_mean;
        let gpu_d = gpu_last_logits[i] - gpu_mean;
        cov += cpu_d * gpu_d;
        cpu_var += cpu_d * cpu_d;
        gpu_var += gpu_d * gpu_d;
    }
    let corr = if cpu_var > 0.0 && gpu_var > 0.0 {
        cov / (cpu_var.sqrt() * gpu_var.sqrt())
    } else {
        0.0
    };
    eprintln!("Correlation: {:.4}", corr);

    Ok(())
}
