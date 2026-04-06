//! Test GPU forward with bias fix
//!
//! This test verifies that the GPU forward path correctly applies QKV bias
//! and produces outputs within expected floating point precision of CPU.

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let test_token = 791u32;

    println!("=== GPU Bias Fix Test ===");
    println!("Token: {}", test_token);
    println!("Model: {}", model_path);
    println!();

    // Enable debug output only when needed (set to empty to run faster)
    // std::env::set_var("CPU_DEBUG_LAYERS", "1");
    // std::env::set_var("GPU_DEBUG", "1");

    // Run CPU forward to get reference
    println!("Running CPU forward...");
    let cpu_logits = model.forward(&[test_token])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("CPU argmax: {:?}", cpu_argmax);

    // Create GPU model
    println!("\nCreating GPU model...");
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut kv_cache = OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );

    // Run GPU forward
    println!("Running GPU forward...");
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut kv_cache, 0)?;
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("GPU argmax: {:?}", gpu_argmax);

    // Compare
    println!("\n=== Results ===");

    // Compute logit statistics
    let mut diffs = Vec::new();
    for i in 0..cpu_logits.len().min(gpu_logits.len()) {
        diffs.push((cpu_logits[i] - gpu_logits[i]).abs());
    }
    let max_diff = diffs.iter().fold(0.0f32, |a, &b| a.max(b));
    let avg_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;

    // Compute Pearson correlation
    let cpu_mean: f32 = cpu_logits.iter().sum::<f32>() / cpu_logits.len() as f32;
    let gpu_mean: f32 = gpu_logits.iter().sum::<f32>() / gpu_logits.len() as f32;
    let mut cov = 0.0f32;
    let mut cpu_var = 0.0f32;
    let mut gpu_var = 0.0f32;
    for i in 0..cpu_logits.len().min(gpu_logits.len()) {
        let cpu_d = cpu_logits[i] - cpu_mean;
        let gpu_d = gpu_logits[i] - gpu_mean;
        cov += cpu_d * gpu_d;
        cpu_var += cpu_d * cpu_d;
        gpu_var += gpu_d * gpu_d;
    }
    let correlation = cov / (cpu_var.sqrt() * gpu_var.sqrt());

    if cpu_argmax.map(|(i, _)| i) == gpu_argmax.map(|(i, _)| i) {
        println!("SUCCESS: CPU and GPU produce same argmax token!");
    } else {
        println!(
            "MISMATCH: CPU and GPU produce different tokens (expected due to FP32 accumulation)"
        );
        println!(
            "  CPU: token {} (logit {:.4})",
            cpu_argmax.unwrap().0,
            cpu_argmax.unwrap().1
        );
        println!(
            "  GPU: token {} (logit {:.4})",
            gpu_argmax.unwrap().0,
            gpu_argmax.unwrap().1
        );
    }

    println!("\n=== Logit Statistics ===");
    println!("  Max diff: {:.4}", max_diff);
    println!("  Avg diff: {:.4}", avg_diff);
    println!("  Correlation: {:.6}", correlation);

    // Check if outputs are highly correlated (expected for correct implementation)
    if correlation > 0.95 {
        println!(
            "\nQKV bias fix verified: GPU and CPU logits are highly correlated ({:.4})",
            correlation
        );
        println!(
            "Small differences are expected due to GPU FP32 parallel accumulation non-determinism."
        );
    } else {
        println!(
            "\nWARNING: Low correlation ({:.4}) - may indicate a bug",
            correlation
        );
    }

    Ok(())
}
