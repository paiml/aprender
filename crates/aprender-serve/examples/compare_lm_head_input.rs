//! Compare CPU vs GPU hidden state before LM head

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let head_dim = hidden_dim / cpu_model.config().num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let test_token: u32 = 791;

    // Run CPU forward to get the hidden state before output norm
    // We need to expose intermediate values. Let's use a custom forward that stops early.
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    // Compare first 20 logits in detail
    eprintln!("=== CPU Logits first 20 ===");
    for logit in cpu_logits.iter().take(20) {
        eprint!("{:.4} ", logit);
    }
    eprintln!();

    // Run GPU
    std::env::set_var("GPU_DEBUG", "1");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    eprintln!("\n=== GPU Logits first 20 ===");
    for logit in gpu_logits.iter().take(20) {
        eprint!("{:.4} ", logit);
    }
    eprintln!();

    // Compute element-wise differences
    eprintln!("\n=== Element-wise diff (GPU - CPU) first 20 ===");
    for i in 0..20 {
        eprint!("{:.4} ", gpu_logits[i] - cpu_logits[i]);
    }
    eprintln!();

    // Stats
    let cpu_sum: f32 = cpu_logits.iter().sum();
    let gpu_sum: f32 = gpu_logits.iter().sum();
    let cpu_mean = cpu_sum / cpu_logits.len() as f32;
    let gpu_mean = gpu_sum / gpu_logits.len() as f32;

    eprintln!("\n=== Stats ===");
    eprintln!("CPU: sum={:.2}, mean={:.4}", cpu_sum, cpu_mean);
    eprintln!("GPU: sum={:.2}, mean={:.4}", gpu_sum, gpu_mean);
    eprintln!(
        "Diff: sum={:.2}, mean={:.4}",
        gpu_sum - cpu_sum,
        gpu_mean - cpu_mean
    );

    // Find the biggest positive and negative differences
    let mut max_diff = (0, 0.0f32);
    let mut min_diff = (0, 0.0f32);
    for i in 0..cpu_logits.len() {
        let diff = gpu_logits[i] - cpu_logits[i];
        if diff > max_diff.1 {
            max_diff = (i, diff);
        }
        if diff < min_diff.1 {
            min_diff = (i, diff);
        }
    }
    eprintln!(
        "\nMax diff: token {} = {:.4} (CPU={:.4}, GPU={:.4})",
        max_diff.0, max_diff.1, cpu_logits[max_diff.0], gpu_logits[max_diff.0]
    );
    eprintln!(
        "Min diff: token {} = {:.4} (CPU={:.4}, GPU={:.4})",
        min_diff.0, min_diff.1, cpu_logits[min_diff.0], gpu_logits[min_diff.0]
    );

    Ok(())
}
