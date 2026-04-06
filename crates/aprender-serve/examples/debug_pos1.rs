//! Debug position 1 to find KV cache issue

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    std::env::set_var("GPU_DEBUG", "1");

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let head_dim = hidden_dim / cpu_model.config().num_heads;
    let kv_dim = num_kv_heads * head_dim;

    // Run position 0 then position 1
    let token0: u32 = 791;
    let token1: u32 = 16;

    eprintln!("=== CPU: Position 0 ===");
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits0 = cpu_model.forward_single_with_cache(token0, &mut cpu_cache, 0)?;
    let cpu_argmax0 = cpu_logits0
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("CPU pos0 argmax: {}", cpu_argmax0);

    eprintln!("\n=== CPU: Position 1 ===");
    let cpu_logits1 = cpu_model.forward_single_with_cache(token1, &mut cpu_cache, 1)?;
    let cpu_argmax1 = cpu_logits1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("CPU pos1 argmax: {}", cpu_argmax1);

    eprintln!("\n=== GPU: Position 0 ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits0 = cuda_model.forward_gpu_resident(token0, &mut gpu_cache, 0)?;
    let gpu_argmax0 = gpu_logits0
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("GPU pos0 argmax: {}", gpu_argmax0);

    eprintln!("\n=== GPU: Position 1 ===");
    let gpu_logits1 = cuda_model.forward_gpu_resident(token1, &mut gpu_cache, 1)?;
    let gpu_argmax1 = gpu_logits1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("GPU pos1 argmax: {}", gpu_argmax1);

    Ok(())
}
