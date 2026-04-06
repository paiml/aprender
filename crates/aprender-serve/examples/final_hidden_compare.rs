//! Compare final hidden state between CPU and GPU before LM head

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let test_token: u32 = 791;

    // CPU forward - get intermediate hidden state from layer outputs
    let cpu_logits = model.forward(&[test_token])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    println!("=== CPU Results ===");
    println!("logits[16] = {:.6}", cpu_logits[16]);
    println!("logits[74403] = {:.6}", cpu_logits[74403]);
    println!("argmax = {} (logit={:.6})", cpu_argmax.0, cpu_argmax.1);

    // GPU forward
    let mapped_gpu = MappedGGUFModel::from_path(path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let hidden_dim = model.config().hidden_dim;
    let kv_dim = model.config().num_kv_heads * (hidden_dim / model.config().num_heads);
    let mut gpu_cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 64);

    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    println!("\n=== GPU Results ===");
    println!("logits[16] = {:.6}", gpu_logits[16]);
    println!("logits[74403] = {:.6}", gpu_logits[74403]);
    println!("argmax = {} (logit={:.6})", gpu_argmax.0, gpu_argmax.1);

    // Compare logits around the critical tokens
    println!("\n=== Comparison ===");
    println!(
        "Diff at [16]: {:.6}",
        (gpu_logits[16] - cpu_logits[16]).abs()
    );
    println!(
        "Diff at [74403]: {:.6}",
        (gpu_logits[74403] - cpu_logits[74403]).abs()
    );

    // Find max difference
    let (max_diff_idx, max_diff) = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .enumerate()
        .map(|(i, (c, g))| (i, (c - g).abs()))
        .max_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
        .unwrap();
    println!(
        "Max diff at idx {}: {:.6} (cpu={:.6}, gpu={:.6})",
        max_diff_idx, max_diff, cpu_logits[max_diff_idx], gpu_logits[max_diff_idx]
    );

    // Correlation
    let cpu_mean: f64 = cpu_logits.iter().map(|&x| x as f64).sum::<f64>() / cpu_logits.len() as f64;
    let gpu_mean: f64 = gpu_logits.iter().map(|&x| x as f64).sum::<f64>() / gpu_logits.len() as f64;
    let mut cov = 0.0f64;
    let mut cpu_var = 0.0f64;
    let mut gpu_var = 0.0f64;
    for (&c, &g) in cpu_logits.iter().zip(&gpu_logits) {
        let c_d = c as f64 - cpu_mean;
        let g_d = g as f64 - gpu_mean;
        cov += c_d * g_d;
        cpu_var += c_d * c_d;
        gpu_var += g_d * g_d;
    }
    let corr = cov / (cpu_var.sqrt() * gpu_var.sqrt());
    println!("Correlation: {:.6}", corr);

    Ok(())
}
