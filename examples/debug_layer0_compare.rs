//! Compare layer 0 output between CPU and GPU

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");
    std::env::set_var("DEBUG_LAYER0", "1");

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

    // CPU layer 0 debug
    eprintln!("=== CPU Layer 0 ===");
    let _cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    // Get embedding
    let embed = cpu_model.embed(&[test_token]);
    let embed_sum: f32 = embed.iter().sum();
    eprintln!(
        "CPU embed sum: {:.6}, first 5: {:?}",
        embed_sum,
        &embed[..5]
    );

    // GPU layer 0 debug
    eprintln!("\n=== GPU Layer 0 ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // Run forward to get logits
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let _gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    Ok(())
}
