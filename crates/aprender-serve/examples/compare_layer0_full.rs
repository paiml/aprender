//! Compare CPU vs GPU full layer 0 step by step

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
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let num_layers = cpu_model.config().num_layers;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let _q_dim = num_heads * head_dim;

    eprintln!(
        "Config: hidden={}, heads={}, kv_heads={}, head_dim={}",
        hidden_dim, num_heads, num_kv_heads, head_dim
    );

    // Run CPU forward pass and capture layer 0 output
    let test_token: u32 = 791;
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    // We need access to intermediate values. Let's get the CPU layer 0 attention output.
    // The CPU model exposes forward_single_layer for debugging:
    let embedding = cpu_model.embed(&[test_token]);
    eprintln!("\nEmbedding first 5: {:?}", &embedding[..5]);

    // Get CPU forward output - we can check intermediate values via debug prints
    let logits_cpu = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;
    let cpu_argmax = logits_cpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] argmax={}, logits[argmax]={:.4}",
        cpu_argmax, logits_cpu[cpu_argmax]
    );
    eprintln!("[CPU] logits first 10: {:?}", &logits_cpu[..10]);

    // Run GPU forward pass
    std::env::set_var("GPU_DEBUG", "1");

    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits_gpu = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;
    let gpu_argmax = logits_gpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "\n[GPU] argmax={}, logits[argmax]={:.4}",
        gpu_argmax, logits_gpu[gpu_argmax]
    );
    eprintln!("[GPU] logits first 10: {:?}", &logits_gpu[..10]);

    // Compare specific positions
    eprintln!("\n=== Comparison ===");
    eprintln!(
        "At CPU argmax ({}): CPU={:.4}, GPU={:.4}",
        cpu_argmax, logits_cpu[cpu_argmax], logits_gpu[cpu_argmax]
    );
    eprintln!(
        "At GPU argmax ({}): CPU={:.4}, GPU={:.4}",
        gpu_argmax, logits_cpu[gpu_argmax], logits_gpu[gpu_argmax]
    );

    // Check correlation
    let n = logits_cpu.len();
    let cpu_mean: f32 = logits_cpu.iter().sum::<f32>() / n as f32;
    let gpu_mean: f32 = logits_gpu.iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut cpu_var = 0.0f32;
    let mut gpu_var = 0.0f32;
    for i in 0..n {
        let cpu_d = logits_cpu[i] - cpu_mean;
        let gpu_d = logits_gpu[i] - gpu_mean;
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

    if corr > 0.99 {
        eprintln!("\n✅ CPU and GPU match closely!");
    } else {
        eprintln!("\n❌ MISMATCH detected. Divergence in forward pass.");
    }

    Ok(())
}
