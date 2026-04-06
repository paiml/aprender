//! Compare CPU vs GPU Q projection for layer 0
//!
//! Run: CUDA_GRAPH_DISABLE=1 GPU_DEBUG=1 cargo run --release --features cuda --example compare_q_projection

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
    let q_dim = num_heads * head_dim; // = hidden_dim
    let eps = cpu_model.config().eps;

    eprintln!("\nModel config:");
    eprintln!("  hidden_dim: {}", hidden_dim);
    eprintln!("  num_heads: {}", num_heads);
    eprintln!("  num_kv_heads: {}", num_kv_heads);
    eprintln!("  head_dim: {}", head_dim);
    eprintln!("  q_dim: {}", q_dim);
    eprintln!("  kv_dim: {}", kv_dim);

    // Test token
    let test_token: u32 = 791;

    // Get embedding
    let embedding = cpu_model.embed(&[test_token]);
    eprintln!("\nEmbedding first 5: {:?}", &embedding[..5]);

    // CPU: RMSNorm
    let layer = &cpu_model.layers()[0];
    let ss: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (ss / hidden_dim as f32 + eps).sqrt();
    let cpu_normed: Vec<f32> = embedding
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| (x / rms) * w)
        .collect();
    eprintln!("\n=== CPU RMSNorm ===");
    eprintln!("RMS: {:.6}", rms);
    eprintln!("Normed first 5: {:?}", &cpu_normed[..5]);

    // CPU: Full QKV matmul
    let qkv_cpu = cpu_model.qkv_matmul(&cpu_normed, &layer.qkv_weight)?;
    eprintln!("\n=== CPU QKV projection ===");
    eprintln!("QKV len: {}", qkv_cpu.len());
    eprintln!("Q (first 5): {:?}", &qkv_cpu[..5]);
    eprintln!("K (first 5): {:?}", &qkv_cpu[q_dim..q_dim + 5]);
    eprintln!(
        "V (first 5): {:?}",
        &qkv_cpu[q_dim + kv_dim..q_dim + kv_dim + 5]
    );

    // At position 0, RoPE is identity (cos=1, sin=0), so we skip it
    let q_cpu = &qkv_cpu[..q_dim];
    let k_cpu = &qkv_cpu[q_dim..q_dim + kv_dim];
    let v_cpu = &qkv_cpu[q_dim + kv_dim..];

    eprintln!("\n=== CPU Q/K/V (before RoPE, pos=0 would be identity anyway) ===");
    eprintln!("Q first 5: {:?}", &q_cpu[..5]);
    eprintln!("K first 5: {:?}", &k_cpu[..5]);
    eprintln!("V first 5: {:?}", &v_cpu[..5]);

    // GPU forward - use full forward but add debug output
    eprintln!("\n=== GPU Path ===");
    std::env::set_var("GPU_DEBUG", "1");

    // Load for GPU
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;

    // Preload weights
    let _ = cuda_model.preload_weights_gpu()?;

    // Run one forward pass - debug output should show intermediate values
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits_gpu = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    let gpu_sum: f32 = logits_gpu.iter().sum();
    let gpu_argmax = logits_gpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("\n=== GPU Logits ===");
    eprintln!("Logits sum: {:.4}, argmax: {}", gpu_sum, gpu_argmax);

    // CPU forward for comparison
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits_cpu = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    let cpu_sum: f32 = logits_cpu.iter().sum();
    let cpu_argmax = logits_cpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("\n=== CPU Logits ===");
    eprintln!("Logits sum: {:.4}, argmax: {}", cpu_sum, cpu_argmax);

    // Compare
    eprintln!("\n=== Comparison ===");
    if cpu_argmax == gpu_argmax {
        eprintln!("CPU and GPU agree on argmax: {}", cpu_argmax);
    } else {
        eprintln!(
            "MISMATCH: CPU argmax={}, GPU argmax={}",
            cpu_argmax, gpu_argmax
        );
        eprintln!("CPU logit at {}: {:.4}", cpu_argmax, logits_cpu[cpu_argmax]);
        eprintln!("GPU logit at {}: {:.4}", gpu_argmax, logits_gpu[gpu_argmax]);
    }

    Ok(())
}
