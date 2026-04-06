//! Trace layer-by-layer hidden state to find where offset originates

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");
    std::env::set_var("GPU_DEBUG", "1");

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

    // GPU: Run forward and capture intermediate states via debug flags
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // Run with LAYER_DEBUG environment to trace
    eprintln!("=== Running GPU forward with layer tracing ===");
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    // Just check final logits stats
    let gpu_mean: f32 = gpu_logits.iter().sum::<f32>() / gpu_logits.len() as f32;
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("\nGPU final logits mean: {:.6}", gpu_mean);
    eprintln!(
        "GPU argmax: {} (logit: {:.4})",
        gpu_argmax, gpu_logits[gpu_argmax]
    );

    Ok(())
}
