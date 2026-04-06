//! Step-by-step comparison of layer 0 between CPU and GPU

use realizar::gguf::{
    MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedModel,
    OwnedQuantizedModelCuda,
};
use realizar::quantize::fused_q4k_parallel_matvec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;
    let test_token: u32 = 791;

    // Step 1: Embedding (should match)
    let embedding = model.embed(&[test_token]);
    println!("=== Step 1: Embedding ===");
    println!("  first 5: {:?}", &embedding[..5]);
    println!("  sum: {:.6}", embedding.iter().sum::<f32>());

    // Step 2: RMSNorm (should match - same CPU implementation)
    let gamma = &model.layers()[0].attn_norm_weight;
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = embedding
        .iter()
        .zip(gamma)
        .map(|(x, g)| x / rms * g)
        .collect();
    println!("\n=== Step 2: RMSNorm ===");
    println!("  first 5: {:?}", &normed[..5]);
    println!("  sum: {:.6}", normed.iter().sum::<f32>());

    // Step 3: Q projection using fused_q4k_parallel_matvec (verified correct)
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: _k_weight,
        v: _v_weight,
    } = &model.layers()[0].qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    let q_cpu = fused_q4k_parallel_matvec(&q_weight.data, &normed, hidden_dim, q_weight.out_dim)?;
    println!("\n=== Step 3: Q projection (CPU fused_q4k_parallel_matvec) ===");
    println!("  first 5: {:?}", &q_cpu[..5]);
    println!("  sum: {:.6}", q_cpu.iter().sum::<f32>());

    // Now compare with GPU path
    println!("\n=== GPU Path ===");
    let mapped_gpu = MappedGGUFModel::from_path(path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // Get GPU Q projection output
    let executor = cuda_model.executor_mut();
    let mut gpu_q = vec![0.0f32; q_weight.out_dim];

    executor.q4k_gemv_cached_tiled(
        "blk.0.attn_q.weight",
        &normed,
        &mut gpu_q,
        q_weight.out_dim as u32,
        hidden_dim as u32,
    )?;

    println!("GPU Q projection:");
    println!("  first 5: {:?}", &gpu_q[..5]);
    println!("  sum: {:.6}", gpu_q.iter().sum::<f32>());

    // Compare
    let max_diff = q_cpu
        .iter()
        .zip(&gpu_q)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\nQ CPU vs GPU max diff: {:.6}", max_diff);

    // Now get actual CPU qkv_matmul output
    println!("\n=== CPU model.qkv_matmul() ===");
    let cpu_qkv = model.qkv_matmul(&normed, &model.layers()[0].qkv_weight)?;
    let cpu_q_from_qkv = &cpu_qkv[0..q_weight.out_dim];
    println!("  first 5: {:?}", &cpu_q_from_qkv[..5]);
    println!("  sum: {:.6}", cpu_q_from_qkv.iter().sum::<f32>());

    let max_diff_qkv = q_cpu
        .iter()
        .zip(cpu_q_from_qkv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  vs fused_q4k_parallel_matvec: max diff = {:.6}",
        max_diff_qkv
    );

    // Full forward CPU
    println!("\n=== Full CPU forward ===");
    let cpu_logits = model.forward(&[test_token])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("  argmax: {}", cpu_argmax);

    // Full forward GPU
    println!("\n=== Full GPU forward ===");
    let kv_dim = model.config().num_kv_heads * (hidden_dim / model.config().num_heads);
    let mut gpu_cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("  argmax: {}", gpu_argmax);

    println!(
        "\nResult: CPU argmax={}, GPU argmax={}",
        cpu_argmax, gpu_argmax
    );
    if cpu_argmax == gpu_argmax {
        println!("PASS");
    } else {
        println!("FAIL - outputs differ!");
    }

    Ok(())
}
