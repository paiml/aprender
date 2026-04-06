//! Compare hidden state before LM head between CPU and GPU

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1"); // Disable graphs for cleaner comparison

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

    // CPU forward - capture hidden state before LM head
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    // Get CPU hidden state by working backwards (this is a diagnostic hack)
    eprintln!("CPU first 10 logits: {:?}", &cpu_logits[..10]);

    // GPU forward
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    eprintln!("GPU first 10 logits: {:?}", &gpu_logits[..10]);

    // Compare logits statistics
    let cpu_mean: f32 = cpu_logits.iter().sum::<f32>() / cpu_logits.len() as f32;
    let gpu_mean: f32 = gpu_logits.iter().sum::<f32>() / gpu_logits.len() as f32;

    eprintln!("\nCPU logits mean: {:.6}", cpu_mean);
    eprintln!("GPU logits mean: {:.6}", gpu_mean);
    eprintln!("Mean difference: {:.6}", gpu_mean - cpu_mean);

    // The mean difference tells us about systematic bias
    // If it's near zero, the bias is random. If it's positive/negative, there's systematic offset.

    // Check variance
    let cpu_var: f32 = cpu_logits
        .iter()
        .map(|x| (x - cpu_mean).powi(2))
        .sum::<f32>()
        / cpu_logits.len() as f32;
    let gpu_var: f32 = gpu_logits
        .iter()
        .map(|x| (x - gpu_mean).powi(2))
        .sum::<f32>()
        / gpu_logits.len() as f32;

    eprintln!("\nCPU logits std: {:.6}", cpu_var.sqrt());
    eprintln!("GPU logits std: {:.6}", gpu_var.sqrt());

    // Correlation
    let n = cpu_logits.len();
    let mut cov = 0.0f32;
    for i in 0..n {
        cov += (cpu_logits[i] - cpu_mean) * (gpu_logits[i] - gpu_mean);
    }
    let corr = cov / (cpu_var.sqrt() * gpu_var.sqrt() * n as f32);
    eprintln!("\nCorrelation: {:.6}", corr);

    // Check argmax
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    eprintln!(
        "\nCPU argmax: {} (logit: {:.4})",
        cpu_argmax, cpu_logits[cpu_argmax]
    );
    eprintln!(
        "GPU argmax: {} (logit: {:.4})",
        gpu_argmax, gpu_logits[gpu_argmax]
    );

    Ok(())
}
